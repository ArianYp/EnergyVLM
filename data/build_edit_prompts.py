#!/usr/bin/env python3
"""Build the DIV2K and Cat2Dog editing sets with SoftREPA's recipe (arXiv 2503.08250, appendix C).

  images   DIV2K: the 800 DIV2K_train_HR images.  Cat2Dog: the 500 AFHQ val/cat images.
           Resized to 512x512 exactly as SoftREPA's edit_sd3.py does (transforms.Resize((s, s)) on a PIL
           image: bilinear with antialias, aspect ratio NOT preserved), saved as PNG.
  source   LLaVA-1.5-13B, the instruction "Describe the object and background in the image", greedy.
           SoftREPA used the 4-bit community quantisation 4bit/llava-v1.5-13b-3GB; we use the same model in
           bf16 (llava-hf/llava-1.5-13b-hf).
  target   DIV2K: Llama-3.1-8B-Instruct with SoftREPA's instruction verbatim (one object replaced), greedy.
           Weights from unsloth/Llama-3.1-8B-Instruct (an ungated copy of meta-llama/Llama-3.1-8B-Instruct).
           Cat2Dog: SoftREPA does not describe its prompts; we replace the cat words of the source caption
           by dog words (deterministic), so the only edit is the species.

Committed outputs: data/editing/{div2k_set,cat2dog_set}/{records.json,manifest.json} (docs/editing/). With those
present, a run only (re)writes the 512 px images and reuses the committed prompts; delete records.json to rebuild them.

  python data/build_edit_prompts.py --dataset div2k       (scripts/edit_prompts.lsf; needs one GPU for the prompts)
  python data/build_edit_prompts.py --dataset cat2dog
Raw data: DIV2K_HR = the unzipped DIV2K_train_HR/ (800 PNGs, data.vision.ee.ethz.ch/cvl/DIV2K), AFHQ_CAT = afhq/val/cat/
(500 JPGs, StarGAN-v2 AFHQ). Models: LLAVA_MODEL / LLAMA_MODEL (Hub ids or local copies of them).
"""
from __future__ import annotations

import argparse, difflib, hashlib, json, os, re, time
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
SOURCES = {"div2k": (Path(os.environ.get("DIV2K_HR", REPO / "data/editing/raw/DIV2K_train_HR")), "*.png"),
           "cat2dog": (Path(os.environ.get("AFHQ_CAT", REPO / "data/editing/raw/afhq/val/cat")), "*.jpg")}
LLAVA = os.environ.get("LLAVA_MODEL", "llava-hf/llava-1.5-13b-hf")
LLAMA = os.environ.get("LLAMA_MODEL", "unsloth/Llama-3.1-8B-Instruct")
CAPTION_INSTRUCTION = "Describe the object and background in the image"

# SoftREPA appendix C, verbatim.
TARGET_INSTRUCTION = """You are an AI assistant for generating paired text prompts for real image editing tasks. Your goal is to modify a given text description by replacing an object with other while strictly following these rules:
• 1) Modify only one object (i.e., a single meaningful concept such as an object). It could be small object.
• 2) The replacement must be significantly different from the original concept but contextually appropriate. Avoid unrealistic substitutions (e.g., changing "rabbit on grass" to "rocket on grass").
• 3) Ensure diversity in word choices across different modifications.
• 4) Preserve all other words exactly as they are. Do not change sentence structure, introduce new elements, or modify additional details.
• 5) Do not provide any additional words—output only the modified text description.
• 6) Do not change or add colors. Specifically, when modifying a building, change only the appearance, not the type of building (e.g., do not change "building" to "church" or "lighthouse").
• 7) Modify only one feature at a time. If changing an object (e.g., "starfish" to "sea turtle"), do not alter its color, shape, or other attributes.
Example:
Input: The image features a close-up of a brown dog with a blue nose. The dog is standing in a grassy field, and the background is blurred, creating a focus on the dog's face. The dog's ears are perked up, and its eyes are open, giving it a curious and attentive expression. The dog's fur is brown, and the grass in the background is green, creating a natural and vibrant scene.
Output: The image features a close-up of a brown fox with a blue nose. The fox is standing in a grassy field, and the background is blurred, creating a focus on the fox's face. The fox's ears are perked up, and its eyes are open, giving it a curious and attentive expression. The fox's fur is brown, and the grass in the background is green, creating a natural and vibrant scene."""

CAT2DOG = [(r"\bkittens\b", "puppies"), (r"\bkitten\b", "puppy"), (r"\bkitties\b", "puppies"),
           (r"\bkitty\b", "puppy"), (r"\bcats\b", "dogs"), (r"\bcat's\b", "dog's"), (r"\bcat\b", "dog"),
           (r"\bfelines\b", "dogs"), (r"\bfeline\b", "canine")]


# LLaVA sometimes names the AFHQ cat a big cat; used only when no cat word is present (1 of 500 captions).
BIGCAT2DOG = [(r"\btigers\b", "dogs"), (r"\btiger's\b", "dog's"), (r"\btiger\b", "dog"),
              (r"\blions\b", "dogs"), (r"\blion's\b", "dog's"), (r"\blion\b", "dog"),
              (r"\bleopards\b", "dogs"), (r"\bleopard's\b", "dog's"), (r"\bleopard\b", "dog"),
              (r"\blynx\b", "dog")]


def _sub(text: str, table) -> str:
    for pat, rep in table:
        text = re.sub(pat, lambda m, r=rep: r.capitalize() if m.group(0)[0].isupper() else r, text,
                      flags=re.IGNORECASE)
    return text


def cat_to_dog(text: str) -> str:
    out = _sub(text, CAT2DOG)
    return out if out != text else _sub(text, BIGCAT2DOG)


def changed_words(a: str, b: str) -> int:
    sm = difflib.SequenceMatcher(a=a.split(), b=b.split())
    return sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal")


def prepare_images(ds: str, out: Path, size: int) -> list[dict]:
    src_dir, pat = SOURCES[ds]
    files = sorted(src_dir.glob(pat))
    expect = {"div2k": 800, "cat2dog": 500}[ds]
    assert len(files) == expect, f"{ds}: {len(files)} images, expected {expect}"
    from torchvision import transforms
    tf = transforms.Resize((size, size))                 # SoftREPA: Resize((s, s)) on PIL
    (out / "images").mkdir(parents=True, exist_ok=True)
    recs = []
    for f in files:
        rid = f.stem
        dst = out / "images" / f"{rid}.png"
        if not dst.exists():
            tf(Image.open(f).convert("RGB")).save(dst)
        recs.append({"id": rid, "image": f"images/{rid}.png", "source_file": str(f),
                     "source_sha256": hashlib.sha256(f.read_bytes()).hexdigest()[:16]})
    return recs


def caption(recs, out: Path, bs: int):
    import torch
    torch.set_grad_enabled(False)
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    proc = AutoProcessor.from_pretrained(LLAVA)
    proc.tokenizer.padding_side = "left"
    model = LlavaForConditionalGeneration.from_pretrained(LLAVA, torch_dtype=torch.bfloat16).cuda().eval()
    prompt = f"USER: <image>\n{CAPTION_INSTRUCTION} ASSISTANT:"
    todo = [r for r in recs if "source_prompt" not in r]
    t0 = time.time()
    for i in range(0, len(todo), bs):
        chunk = todo[i:i + bs]
        imgs = [Image.open(out / r["image"]).convert("RGB") for r in chunk]
        enc = proc(images=imgs, text=[prompt] * len(chunk), return_tensors="pt", padding=True).to("cuda")
        enc["pixel_values"] = enc["pixel_values"].to(torch.bfloat16)
        gen = model.generate(**enc, do_sample=False, num_beams=1, max_new_tokens=300)
        texts = proc.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for r, t in zip(chunk, texts):
            r["source_prompt"] = " ".join(t.strip().split())
        print(f"[caption] {i + len(chunk)}/{len(todo)}  {time.time() - t0:.0f}s  e.g. {chunk[0]['source_prompt'][:90]!r}",
              flush=True)
    del model
    torch.cuda.empty_cache()


def retarget_llama(recs, bs: int):
    import torch
    torch.set_grad_enabled(False)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(LLAMA)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(LLAMA, torch_dtype=torch.bfloat16).cuda().eval()
    todo = [r for r in recs if "target_prompt" not in r]
    t0 = time.time()
    for i in range(0, len(todo), bs):
        chunk = todo[i:i + bs]
        chats = [tok.apply_chat_template(
            [{"role": "system", "content": TARGET_INSTRUCTION},
             {"role": "user", "content": f"Input: {r['source_prompt']}\nOutput:"}],
            tokenize=False, add_generation_prompt=True) for r in chunk]
        enc = tok(chats, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        gen = model.generate(**enc, do_sample=False, num_beams=1, max_new_tokens=400,
                             pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for r, t in zip(chunk, texts):
            t = t.strip()
            t = re.sub(r"^(Output:\s*)", "", t).strip()
            r["target_prompt"] = " ".join(t.split())
            r["target_generator"] = "llama-3.1-8b-instruct"
        print(f"[target] {i + len(chunk)}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)


def needs_repair(r) -> bool:
    """Greedy Llama output that is not a one-object edit: unchanged, or cut to the first sentence(s)."""
    ls, lt = len(r["source_prompt"].split()), len(r["target_prompt"].split())
    return r["n_changed_words"] == 0 or lt < 0.9 * ls


def repair_llama(recs, bs: int, tries: int = 8):
    """Re-ask Llama (same instruction) for the records needs_repair() flags, with seeded sampling
    (temperature 0.7, top_p 0.9); keep the first output that changes 1..6 words at >= 0.9 of the source
    length. Records that never pass keep the greedy output and are flagged. Deterministic given the seeds."""
    import torch
    torch.set_grad_enabled(False)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(LLAMA)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(LLAMA, torch_dtype=torch.bfloat16).cuda().eval()
    todo = [r for r in recs if needs_repair(r)]
    print(f"[repair] {len(todo)} records need repair", flush=True)
    for t in range(tries):
        todo = [r for r in todo if needs_repair(r)]
        if not todo:
            break
        torch.manual_seed(1000 + t)
        for i in range(0, len(todo), bs):
            chunk = todo[i:i + bs]
            chats = [tok.apply_chat_template(
                [{"role": "system", "content": TARGET_INSTRUCTION},
                 {"role": "user", "content": f"Input: {r['source_prompt']}\nOutput:"}],
                tokenize=False, add_generation_prompt=True) for r in chunk]
            enc = tok(chats, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            gen = model.generate(**enc, do_sample=True, temperature=0.7, top_p=0.9, max_new_tokens=400,
                                 pad_token_id=tok.pad_token_id)
            texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for r, txt in zip(chunk, texts):
                cand = " ".join(re.sub(r"^(Output:\s*)", "", txt.strip()).split())
                n = changed_words(r["source_prompt"], cand)
                if 1 <= n <= 6 and len(cand.split()) >= 0.9 * len(r["source_prompt"].split()):
                    r.setdefault("target_prompt_greedy", r["target_prompt"])
                    r["target_prompt"], r["n_changed_words"] = cand, n
                    r["target_repaired"] = f"sampled, try {t}, seed {1000 + t}"
        print(f"[repair] try {t}: {sum(needs_repair(r) for r in recs)} still flagged", flush=True)
    for r in recs:
        if needs_repair(r):
            r["target_repaired"] = "FAILED: greedy output kept"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(SOURCES), required=True)
    ap.add_argument("--out", type=Path, default=None, help="default data/editing/<dataset>_set")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--bs", type=int, default=16)
    a = ap.parse_args()
    out = a.out or REPO / f"data/editing/{a.dataset}_set"
    out.mkdir(parents=True, exist_ok=True)
    mp = out / "records.json"
    fresh = prepare_images(a.dataset, out, a.size)          # always (re)writes missing 512 px images
    if mp.exists():                                         # committed prompts: keep them, check the images match
        recs = json.loads(mp.read_text())
        assert [r["id"] for r in recs] == [r["id"] for r in fresh], "records.json ids differ from the raw images"
        bad = [r["id"] for r, f in zip(recs, fresh) if r.get("source_sha256") != f["source_sha256"]]
        assert not bad, f"raw images differ from the ones the committed prompts describe: {bad[:3]}"
    else:
        recs = fresh
    mp.write_text(json.dumps(recs, indent=1))
    if any("source_prompt" not in r for r in recs):
        caption(recs, out, a.bs)
        mp.write_text(json.dumps(recs, indent=1))
    if a.dataset == "cat2dog":
        for r in recs:
            r["target_prompt"] = cat_to_dog(r["source_prompt"])
            r["target_generator"] = "cat->dog word substitution"
    elif any("target_prompt" not in r for r in recs):
        retarget_llama(recs, a.bs)
    for r in recs:
        r["n_changed_words"] = changed_words(r["source_prompt"], r["target_prompt"])
    if a.dataset == "div2k" and any(needs_repair(r) and "target_repaired" not in r for r in recs):
        repair_llama(recs, a.bs)
    mp.write_text(json.dumps(recs, indent=1))
    n = len(recs)
    same = sum(r["n_changed_words"] == 0 for r in recs)
    big = sum(r["n_changed_words"] > 6 for r in recs)
    manifest = {"dataset": a.dataset, "n": n, "size": a.size, "caption_model": str(LLAVA),
                "caption_instruction": CAPTION_INSTRUCTION,
                "target": "llama" if a.dataset == "div2k" else "cat->dog substitution",
                "target_model": str(LLAMA) if a.dataset == "div2k" else None,
                "unchanged_targets": same, "targets_changing_more_than_6_words": big,
                "targets_repaired_by_sampling": sum(str(r.get("target_repaired", "")).startswith("sampled")
                                                    for r in recs),
                "targets_repair_failed": sum(str(r.get("target_repaired", "")).startswith("FAILED")
                                             for r in recs)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
