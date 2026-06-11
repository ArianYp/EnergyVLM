#!/usr/bin/env python3
"""
Score a directory of DiT-generated images with CLIPScore + VQAScore + DA-Score.

Consumes the output of `generate_dit.py`:
  {eval_dir}/prompt_to_image.json
  {eval_dir}/images/*.png

Produces a unified summary:
  {eval_dir}/summary.json
  {eval_dir}/clipscore_per_prompt.jsonl
  {eval_dir}/vqascore_per_prompt.jsonl
  {eval_dir}/da_score_per_prompt.jsonl
  {eval_dir}/da_score_questions_cache.json       # GPT-4 decompositions (cached)

Single-process, single-GPU. Scorers are loaded sequentially; each is freed
before the next is loaded so a 40 GB GPU fits CLIP-FlanT5-XXL plus BLIP-VQA.

Assumptions:
  - `t2v_metrics`, `transformers`, `openai` are all importable in the active env.
  - `OPENAI_API_KEY` is set in the environment (DA-Score only).
"""

import argparse
import json
import re
from pathlib import Path
from statistics import mean, pstdev

# Compat shim: t2v_metrics ships its own vendored `lavis` copy that imports
# several helpers from transformers.modeling_utils. HF moved some helpers to
# transformers.pytorch_utils (4.x) and removed find_pruneable_heads_and_indices
# entirely by 5.x. Patch what we can, vendor a stub for what was removed, so
# t2v_metrics' lazy imports succeed at `import t2v_metrics` time.
import torch as _torch
import transformers.modeling_utils as _mu
import transformers.pytorch_utils as _pu
for _name in ("apply_chunking_to_forward", "prune_linear_layer"):
    if not hasattr(_mu, _name) and hasattr(_pu, _name):
        setattr(_mu, _name, getattr(_pu, _name))

if not hasattr(_mu, "find_pruneable_heads_and_indices"):
    # Verbatim reimpl of the old HF helper. Only used by pruning code paths
    # that VQAScore inference never touches; this just satisfies the import.
    def _find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
        mask = _torch.ones(n_heads, head_size)
        heads = set(heads) - already_pruned_heads
        for head in heads:
            head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = _torch.arange(len(mask))[mask].long()
        return heads, index
    _mu.find_pruneable_heads_and_indices = _find_pruneable_heads_and_indices
    if not hasattr(_pu, "find_pruneable_heads_and_indices"):
        _pu.find_pruneable_heads_and_indices = _find_pruneable_heads_and_indices

# Some lavis models (e.g. the blip2_t5 path we don't actually use) import
# transformers.utils.model_parallel_utils, a module removed in transformers 5.x.
# We don't need the functionality; install a sys.modules stub so the import line
# succeeds. `assert_device_map` / `get_device_map` are no-ops for our case.
import sys as _sys
import types as _types
if "transformers.utils.model_parallel_utils" not in _sys.modules:
    _stub = _types.ModuleType("transformers.utils.model_parallel_utils")
    _stub.assert_device_map = lambda *a, **kw: None
    _stub.get_device_map = lambda *a, **kw: {}
    _sys.modules["transformers.utils.model_parallel_utils"] = _stub

# t2v_metrics unconditionally imports a PerceptionLM video model that depends
# on `torchcodec` (FFmpeg-backed video decoding). The env has torchcodec but
# no system FFmpeg, so `import torchcodec` fails. We don't score any video here
# (image-only CLIP-FlanT5), so stub it in sys.modules before t2v_metrics loads.
if "torchcodec" not in _sys.modules:
    _tc = _types.ModuleType("torchcodec")
    _tc_decoders = _types.ModuleType("torchcodec.decoders")
    class _UnusableVideoDecoder:  # noqa: D401
        """Stub; video scoring is not used by score_dit.py."""
        def __init__(self, *a, **kw):
            raise RuntimeError("torchcodec stub: video scoring is disabled in score_dit.py")
    _tc_decoders.VideoDecoder = _UnusableVideoDecoder
    _tc.decoders = _tc_decoders
    _sys.modules["torchcodec"] = _tc
    _sys.modules["torchcodec.decoders"] = _tc_decoders

# t2v_metrics' package __init__ eagerly loads every model family, including
# PerceptionLM which depends on torch.nn.attention.flex_attention (torch>=2.5;
# our env has 2.4.1). We only need CLIP-FlanT5 for VQAScore; stub the whole
# PerceptionLM module so its import does not run.
_pl_stub = _types.ModuleType("t2v_metrics.models.vqascore_models.perceptionlm_model")
_pl_stub.PERCEPTION_LM_MODELS = {}
class _UnusablePerceptionLMModel:  # noqa: D401
    """Stub; PerceptionLM is not used by score_dit.py."""
    def __init__(self, *a, **kw):
        raise RuntimeError("PerceptionLM stub: disabled in score_dit.py")
_pl_stub.PerceptionLMModel = _UnusablePerceptionLMModel
_sys.modules["t2v_metrics.models.vqascore_models.perceptionlm_model"] = _pl_stub

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


def safe_mean(xs):
    return float(mean(xs)) if xs else None


def safe_std(xs):
    if not xs:
        return None
    if len(xs) == 1:
        return 0.0
    return float(pstdev(xs))


# ---------------------------------------------------------------------------
# CLIPScore (openai/clip-vit-large-patch14)
# ---------------------------------------------------------------------------

def run_clipscore(prompts, image_paths, clip_score_id, device, batch_size):
    from transformers import CLIPModel, CLIPProcessor
    print(f"  Loading CLIP scorer: {clip_score_id}")
    processor = CLIPProcessor.from_pretrained(clip_score_id)
    model = CLIPModel.from_pretrained(clip_score_id).to(device).eval()

    cosines, scores = [], []
    for start in tqdm(range(0, len(prompts), batch_size), desc="CLIPScore"):
        bp = prompts[start:start + batch_size]
        bi = [Image.open(p).convert("RGB") for p in image_paths[start:start + batch_size]]
        inputs = processor(text=bp, images=bi, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            img_f = F.normalize(model.get_image_features(pixel_values=inputs["pixel_values"]), dim=-1)
            txt_f = F.normalize(
                model.get_text_features(
                    input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
                ),
                dim=-1,
            )
            cos = (img_f * txt_f).sum(dim=-1).cpu().tolist()
        for c in cos:
            cosines.append(float(c))
            scores.append(float(100.0 * max(c, 0.0)))

    del model, processor
    torch.cuda.empty_cache()
    return cosines, scores


# ---------------------------------------------------------------------------
# VQAScore (t2v_metrics, clip-flant5-xxl by default)
# ---------------------------------------------------------------------------

def run_vqascore(prompts, image_paths, model_name, device):
    import t2v_metrics
    print(f"  Loading VQAScore model: {model_name}")
    # t2v_metrics picks the first CUDA device by default. Scope via CUDA_VISIBLE_DEVICES upstream.
    scorer = t2v_metrics.VQAScore(model=model_name)

    scores = []
    # Per-pair call: simple, avoids the M*N matrix blowup of passing matched pairs
    # through the default __call__. ~0.1-0.5 s/pair on an A100 for clip-flant5-xxl.
    for prompt, img_path in tqdm(list(zip(prompts, image_paths)), desc=f"VQAScore({model_name})"):
        with torch.no_grad():
            out = scorer(images=[img_path], texts=[prompt])
        # Out is typically (1, 1) or (1,). Extract scalar robustly.
        if torch.is_tensor(out):
            val = float(out.view(-1)[0].item())
        else:
            val = float(out)
        scores.append(val)

    del scorer
    torch.cuda.empty_cache()
    return scores


# ---------------------------------------------------------------------------
# DA-Score (Decompositional Alignment Score, Singh & Zheng NeurIPS 2023)
#
# Two-stage:
#   1. GPT-4 Turbo decomposes each prompt into N atomic yes/no questions.
#   2. BLIP-VQA (HF transformers) answers each question on the image; we
#      read P(yes) from the first generated-token softmax.
#   DA-Score = mean over questions of P(yes).
#
# The GPT-4 stage is cached to disk so a retry never re-pays OpenAI.
# ---------------------------------------------------------------------------

DA_SYSTEM_PROMPT = (
    "You decompose text-to-image prompts into atomic yes/no verification questions. "
    "Each question must test exactly ONE disjoint assertion: an object's presence, an "
    "attribute (color, material, size), a count, a spatial relation, or an action/verb. "
    "Output ONLY a JSON list of questions (strings), nothing else. No prose, no markdown."
)


def gpt4_decompose(client, prompt, model_name):
    user = (
        f'Decompose the following text-to-image prompt into 2-8 atomic yes/no '
        f'questions that verify whether a generated image contains every '
        f'assertion in the prompt.\n\nPrompt: "{prompt}"\n\nJSON list:'
    )
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": DA_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    text = resp.choices[0].message.content.strip()
    # Try strict JSON first, then fall back to regex-extracted JSON list.
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if not m:
            raise ValueError(f"could not extract JSON list from GPT-4 output: {text[:200]}")
        parsed = json.loads(m.group(0))
    if not isinstance(parsed, list) or not all(isinstance(q, str) for q in parsed):
        raise ValueError(f"GPT-4 output is not a list[str]: {parsed}")
    return parsed


def blip_yes_prob(blip_model, blip_processor, image, question, device):
    """P(yes) from BLIP-VQA, normalized within the {yes, no} pair."""
    inputs = blip_processor(image, question, return_tensors="pt").to(device)
    with torch.no_grad():
        out = blip_model.generate(
            **inputs, max_length=2,
            return_dict_in_generate=True, output_scores=True,
        )
    # out.scores[0] shape: (batch=1, vocab)
    logits = out.scores[0][0]
    probs = F.softmax(logits, dim=-1)
    yes_id = blip_processor.tokenizer.convert_tokens_to_ids("yes")
    no_id = blip_processor.tokenizer.convert_tokens_to_ids("no")
    p_yes = float(probs[yes_id].item())
    p_no = float(probs[no_id].item())
    return p_yes / max(p_yes + p_no, 1e-8)


def run_da_score(prompts, image_paths, cache_path, openai_model, blip_id, device):
    import openai
    client = openai.OpenAI()

    # Stage 1: GPT-4 decomposition, cached.
    cache = {}
    if cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)

    missing = [p for p in prompts if p not in cache]
    if missing:
        print(f"  GPT-4 decomposing {len(missing)} new prompts (cached: {len(cache)})")
    for p in tqdm(missing, desc="DA-Score GPT-4"):
        try:
            cache[p] = gpt4_decompose(client, p, openai_model)
        except Exception as e:
            print(f"  [GPT-4 FAILED on {p[:60]!r}] {e}")
            cache[p] = None
        # Persist after each successful (or failed) call so we resume on crash.
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=2)

    # Stage 2: BLIP-VQA scoring.
    from transformers import BlipForQuestionAnswering, BlipProcessor
    print(f"  Loading BLIP-VQA: {blip_id}")
    blip_model = BlipForQuestionAnswering.from_pretrained(blip_id).to(device).eval()
    blip_processor = BlipProcessor.from_pretrained(blip_id)

    details = []
    for prompt, img_path in tqdm(list(zip(prompts, image_paths)), desc="DA-Score BLIP-VQA"):
        questions = cache.get(prompt)
        if not questions:
            details.append({"prompt": prompt, "questions": None, "yes_probs": None, "da_score": None})
            continue
        image = Image.open(img_path).convert("RGB")
        yes_probs = [
            blip_yes_prob(blip_model, blip_processor, image, q, device) for q in questions
        ]
        da = float(mean(yes_probs)) if yes_probs else None
        details.append({
            "prompt": prompt,
            "questions": questions,
            "yes_probs": yes_probs,
            "da_score": da,
        })

    del blip_model, blip_processor
    torch.cuda.empty_cache()
    return details


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", required=True,
                        help="Dir produced by generate_dit.py (contains prompt_to_image.json + images/).")
    parser.add_argument("--clip-score-id", default="openai/clip-vit-large-patch14")
    parser.add_argument("--vqascore-model", default="clip-flant5-xxl")
    parser.add_argument("--blip-vqa-id", default="Salesforce/blip-vqa-capfilt-large")
    parser.add_argument("--openai-model", default="gpt-4-1106-preview")
    parser.add_argument("--skip-clipscore", action="store_true")
    parser.add_argument("--skip-vqascore", action="store_true")
    parser.add_argument("--skip-da-score", action="store_true")
    parser.add_argument("--clip-batch-size", type=int, default=8)
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(eval_dir / "prompt_to_image.json") as f:
        prompt_to_image = json.load(f)
    prompts = list(prompt_to_image.keys())
    image_paths = [prompt_to_image[p] for p in prompts]
    print(f"Scoring {len(prompts)} (prompt, image) pairs from {eval_dir}")

    with open(eval_dir / "generation_config.json") as f:
        gen_config = json.load(f)

    summary = {
        "eval_dir": str(eval_dir),
        "generation_config": gen_config,
        "scoring_config": {
            "clip_score_id": args.clip_score_id,
            "vqascore_model": args.vqascore_model,
            "blip_vqa_id": args.blip_vqa_id,
            "openai_model": args.openai_model,
        },
        "metrics": {},
    }

    # --- CLIPScore ---
    if not args.skip_clipscore:
        print(f"\n=== CLIPScore ({args.clip_score_id}) ===")
        cosines, scores = run_clipscore(
            prompts, image_paths, args.clip_score_id, device, args.clip_batch_size,
        )
        summary["metrics"]["clipscore"] = {
            "mean": safe_mean(scores),
            "std": safe_std(scores),
            "mean_cosine": safe_mean(cosines),
        }
        with open(eval_dir / "clipscore_per_prompt.jsonl", "w") as f:
            for p, c, s in zip(prompts, cosines, scores):
                f.write(json.dumps({"prompt": p, "clip_cosine": c, "clipscore": s}) + "\n")
        m = summary["metrics"]["clipscore"]
        print(f"  mean CLIPScore   : {m['mean']:.3f} +/- {m['std']:.3f}")
        print(f"  mean CLIP cosine : {m['mean_cosine']:.4f}")

    # --- VQAScore ---
    if not args.skip_vqascore:
        print(f"\n=== VQAScore ({args.vqascore_model}) ===")
        scores = run_vqascore(prompts, image_paths, args.vqascore_model, device)
        summary["metrics"]["vqascore"] = {
            "mean": safe_mean(scores),
            "std": safe_std(scores),
        }
        with open(eval_dir / "vqascore_per_prompt.jsonl", "w") as f:
            for p, s in zip(prompts, scores):
                f.write(json.dumps({"prompt": p, "vqascore": s}) + "\n")
        m = summary["metrics"]["vqascore"]
        print(f"  mean VQAScore    : {m['mean']:.4f} +/- {m['std']:.4f}")

    # --- DA-Score ---
    if not args.skip_da_score:
        print(f"\n=== DA-Score (GPT-4 + BLIP-VQA) ===")
        cache_path = eval_dir / "da_score_questions_cache.json"
        details = run_da_score(
            prompts, image_paths, cache_path,
            args.openai_model, args.blip_vqa_id, device,
        )
        valid = [d["da_score"] for d in details if d["da_score"] is not None]
        summary["metrics"]["da_score"] = {
            "mean": safe_mean(valid),
            "std": safe_std(valid),
            "num_valid": len(valid),
            "num_total": len(details),
        }
        with open(eval_dir / "da_score_per_prompt.jsonl", "w") as f:
            for d in details:
                f.write(json.dumps(d) + "\n")
        m = summary["metrics"]["da_score"]
        print(f"  mean DA-Score    : {m['mean']:.4f} +/- {m['std']:.4f} (valid {m['num_valid']}/{m['num_total']})")

    with open(eval_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n== wrote {eval_dir / 'summary.json'} ==")


if __name__ == "__main__":
    main()
