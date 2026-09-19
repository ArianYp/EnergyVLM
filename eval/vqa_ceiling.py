#!/usr/bin/env python3
"""The ceiling for ANY ranking signal on the structured negatives (the design note's O1 validity).

When the student samples a caption and its rule-based negatives from the SAME noise, do the negatives'
images differ from the positive's in the edited attribute, and does the positive's image describe the
caption better than the negatives' images do? Measured with an external judge (VQAScore) independent of
DINO and of the photograph: for each held-out caption the full matrix S[image i][prompt t] over
{positive, neg_1..neg_m}, and per edit family
    contradiction    P( S[j][0] < S[0][0] )   the negative's image scores lower on the positive prompt
    edit realised    P( S[j][j] > S[0][j] )   the negative's image scores higher on its own prompt
plus the mean gaps and the DINO cosine of each negative's image to the positive's.

Three stages so SD3.5 and clip-flant5-xxl never share a GPU:
    python3 eval/vqa_ceiling.py --stage gen --ckpt checkpoints/<run>/checkpoint_avg_last5.pt --out out/vqa_ceiling
    python3 eval/vqa_ceiling.py --stage score --out out/vqa_ceiling
    python3 eval/vqa_ceiling.py --stage report --out out/vqa_ceiling
Result on the campaign's negatives (docs/rank/debug/vqa_ceiling.md): colour 0.96, texture 0.91, shape 0.85,
verb 0.78, count 0.63, 3d_spatial 0.68 and spatial 0.54 contradiction -- the relation families produce no
usable order, which is why the v3 negatives drop them.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import data.build_negatives as BN  # noqa: E402
from data.build_negatives import negatives_for  # noqa: E402


def load_cache(d):
    out = {}
    for f in sorted(glob.glob(f"{d}/selection_rank*.jsonl")):
        for ln in open(f):
            if ln.strip():
                r = json.loads(ln); out[int(r["idx"])] = (r["prompt"], r["reference"])
    return out


def stage_gen(args, out: Path) -> None:
    from diffusers import StableDiffusion3Pipeline
    from train.rank_utils import rollout_schedule, student_rollout
    from train.distill import PatchScorer
    from common.sampling import vae_decode
    BN.FAMILIES = args.families.split(","); BN.COUNT_MIN_DELTA = args.count_min_delta
    device = torch.device("cuda")
    big, small = load_cache(args.cache), load_cache(args.exclude)
    seen_p = {p for p, _ in small.values()}; seen_r = {r for _, r in small.values()}
    if Path(args.proj_manifest).exists():
        for ln in open(args.proj_manifest):
            if ln.strip():
                r = json.loads(ln); seen_p.add(r["prompt"]); seen_r.add(r["reference"])
    held = sorted(i for i, (p, r) in big.items() if p not in seen_p and r not in seen_r)
    random.Random(args.seed).shuffle(held)
    picks = []
    for i in held:
        negs = negatives_for(big[i][0], args.m)
        if len(negs) >= 2:
            picks.append((i, big[i][0], negs))
        if len(picks) >= args.n:
            break
    pipe = StableDiffusion3Pipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).to(device)
    for mdl in (pipe.vae, pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3):
        mdl.to(dtype=torch.bfloat16).eval()
    if args.ckpt != "base":
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        pipe.transformer.load_state_dict(ck["model"], strict=True)
    model = pipe.transformer.eval()
    sig, ts = rollout_schedule(pipe.scheduler, args.steps, args.K, device)
    scorer = PatchScorer(args.dino_id, device)
    lat_c = model.config.in_channels; H = args.height // pipe.vae_scale_factor
    img_dir = out / "images"; img_dir.mkdir(parents=True, exist_ok=True)
    recs, t0 = [], time.time()
    with torch.no_grad():
        for n_done, (idx, prompt, negs) in enumerate(picks):
            prompts = [prompt] + [x["prompt"] for x in negs]
            emb, _, pooled, _ = pipe.encode_prompt(prompt=prompts, prompt_2=prompts, prompt_3=prompts, do_classifier_free_guidance=False, device=device, num_images_per_prompt=1)
            g = torch.Generator(device=device).manual_seed(int(idx))
            z0 = torch.randn(1, lat_c, H, H, device=device, dtype=torch.bfloat16, generator=g).expand(len(prompts), -1, -1, -1).contiguous()
            zK = student_rollout(model, z0, emb, pooled, sig, ts, grad=False)
            u8 = ((vae_decode(pipe.vae, zK) + 1) / 2 * 255).round().clamp(0, 255).to(torch.uint8)
            paths = []
            for j, x in enumerate(u8):
                p = img_dir / f"p{int(idx):06d}_i{j}.png"; scorer.Image.fromarray(x.permute(1, 2, 0).cpu().numpy()).save(p); paths.append(str(p))
            e = scorer.embed_latents(pipe.vae, zK)
            recs.append({"idx": int(idx), "prompt": prompt, "negatives": negs, "images": paths,
                         "dino_cos_to_pos": (e[1:] * e[:1]).sum(-1).float().cpu().tolist()})
            if n_done % 25 == 0:
                print(f"[gen] {n_done}/{len(picks)} {time.time() - t0:.0f}s", flush=True)
    json.dump({"meta": vars(args), "records": recs}, open(out / "manifest.json", "w"), indent=1)
    print(f"[gen] wrote {out / 'manifest.json'} ({len(recs)} captions)", flush=True)


def stage_score(args, out: Path) -> None:
    man = json.load(open(out / "manifest.json"))
    sys.path.insert(0, args.t2v_dir)
    from common import t2v_compat  # noqa: F401  stubs the API/video backends t2v_metrics imports unconditionally
    import t2v_metrics
    vqa = t2v_metrics.VQAScore(model=args.vqa_model, device="cuda:0")
    scores, t0 = [], time.time()
    with torch.no_grad():
        for n, r in enumerate(man["records"]):
            prompts = [r["prompt"]] + [x["prompt"] for x in r["negatives"]]
            scores.append({"idx": r["idx"], "S": vqa(images=r["images"], texts=prompts).float().cpu().tolist()})
            if n % 25 == 0:
                print(f"[score] {n}/{len(man['records'])} {time.time() - t0:.0f}s", flush=True)
    json.dump({"vqa_model": args.vqa_model, "scores": scores}, open(out / "scores.json", "w"), indent=1)


def stage_report(args, out: Path) -> None:
    man = json.load(open(out / "manifest.json")); sc = {r["idx"]: np.asarray(r["S"]) for r in json.load(open(out / "scores.json"))["scores"]}
    rows = []
    for r in man["records"]:
        S = sc[r["idx"]]
        for j, ng in enumerate(r["negatives"], start=1):
            rows.append({"family": ng["family"], "contra": float(S[j][0] < S[0][0]), "gap_pos": float(S[0][0] - S[j][0]),
                         "realised": float(S[j][j] > S[0][j]), "gap_own": float(S[j][j] - S[0][j]), "dino": r["dino_cos_to_pos"][j - 1]})
    fams = defaultdict(list)
    for x in rows:
        fams[x["family"]].append(x); fams["all"].append(x)
    summary = {f: {"n": len(v), "contradiction": float(np.mean([x["contra"] for x in v])), "gap_pos": float(np.mean([x["gap_pos"] for x in v])),
                   "edit_realised": float(np.mean([x["realised"] for x in v])), "gap_own": float(np.mean([x["gap_own"] for x in v])),
                   "dino_cos_median": float(np.median([x["dino"] for x in v]))} for f, v in fams.items()}
    json.dump(summary, open(out / "vqa_ceiling.json", "w"), indent=1)
    lines = ["| family | n | contradiction | gap (pos prompt) | edit realised | gap (own prompt) | DINO cos median |", "|---|---:|---:|---:|---:|---:|---:|"]
    for f in ["all"] + sorted(k for k in summary if k != "all"):
        s = summary[f]; lines.append(f"| {f} | {s['n']} | {s['contradiction']:.3f} | {s['gap_pos']:+.3f} | {s['edit_realised']:.3f} | {s['gap_own']:+.3f} | {s['dino_cos_median']:.3f} |")
    (out / "vqa_ceiling.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["gen", "score", "report"])
    ap.add_argument("--out", default="out/vqa_ceiling")
    ap.add_argument("--ckpt", default="base")
    ap.add_argument("--cache", default="cache/train"); ap.add_argument("--exclude", default="cache/train_3k")
    ap.add_argument("--proj_manifest", default="cache/latents/manifest.jsonl")
    ap.add_argument("--families", default=",".join(BN.FAMILY_ORDER)); ap.add_argument("--count_min_delta", type=int, default=1)
    ap.add_argument("--n", type=int, default=200); ap.add_argument("--m", type=int, default=3); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=4); ap.add_argument("--K", type=int, default=8); ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--model_id", default="stabilityai/stable-diffusion-3.5-medium"); ap.add_argument("--dino_id", default="facebook/dinov2-base")
    ap.add_argument("--t2v_dir", default=str(ROOT / "third_party" / "t2v_metrics")); ap.add_argument("--vqa_model", default="clip-flant5-xxl")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    {"gen": stage_gen, "score": stage_score, "report": stage_report}[args.stage](args, out)


if __name__ == "__main__":
    main()
