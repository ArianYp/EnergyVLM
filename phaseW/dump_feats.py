#!/usr/bin/env python3
"""Dump the frozen features the projector gate needs: DINO image features + text embeddings.

Nothing is generated here. CompBench candidate images already exist on disk from the candidate
evaluation; COCO candidates are decoded from the cached endpoint latents, which is the same decode
phaseV/score_dino_patch.py used, so the features are comparable to the 39.9% / 43.6% numbers.

Four image representations per image, because the choice matters and we have measured that it does
(mean-pooled patches beat CLS by 3.7 points of headroom on COCO, job 126439):

    dinov2-large  CLS   [1024]      dinov2-large  mean-patch  [1024]
    dinov2-base   CLS   [ 768]      dinov2-base   mean-patch  [ 768]

Text side, taken from SD3.5's own encoders so the head sees what the model is actually conditioned
on: the pooled CLIP-L/CLIP-G projection [2048] and the sequence-mean of the joint T5+CLIP stream
[4096]. Only the `text_head` arm uses these; every prompt-blind arm ignores them.

The CompBench branch recovers the prompt index from the sample filename and CROSS-CHECKS the parsed
prompt string against the cached VQAScore record for that index. A mismatch aborts: an off-by-one in
that join would silently scramble every label in the gate.
"""
from __future__ import annotations

import argparse, glob, json, re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def load_prompts() -> dict[int, str]:
    """idx -> prompt, from the cached CompBench VQAScore records."""
    out = {}
    for f in glob.glob("phaseN/bestofk_rec_113031/compbench_vqa_rank*.jsonl"):
        for line in open(f):
            if line.strip():
                r = json.loads(line); out[int(r["idx"])] = r["prompt"]
    return out


def prompt_to_idx() -> dict[tuple[str, str], int]:
    """(category, prompt text) -> global prompt index.

    The six-digit suffix on a sample filename is a PER-CATEGORY counter, not the global index the
    label and VQAScore artifacts key on, so the join has to go through the prompt string. The
    category qualifier keeps prompts that recur across categories from colliding.
    """
    prompts = load_prompts()
    cat = {}
    r = json.load(open(sorted(glob.glob("phaseX/candeval/cand*/candidate_labels.json"))[0]))
    for k, d in r["compbench"].items():
        for pp in d["per_prompt"]:
            cat[int(pp["idx"])] = k
    out = {}
    for i, p in prompts.items():
        if i not in cat:
            continue
        key = (cat[i], p.strip())
        if key in out:
            raise SystemExit(f"duplicate prompt within category, join is ambiguous: {key}")
        out[key] = i
    return out


class Dino:
    def __init__(self, device):
        from transformers import AutoModel, AutoImageProcessor
        self.m, self.p = {}, {}
        for tag, mid in (("l", "facebook/dinov2-large"), ("b", "facebook/dinov2-base")):
            self.m[tag] = AutoModel.from_pretrained(mid).to(device).eval()
            self.p[tag] = AutoImageProcessor.from_pretrained(mid)
        self.device = device

    @torch.no_grad()
    def __call__(self, images):
        out = {}
        for tag in ("l", "b"):
            px = self.p[tag](images=images, return_tensors="pt")["pixel_values"].to(self.device)
            h = self.m[tag](pixel_values=px).last_hidden_state
            out[f"{tag}_cls"] = F.normalize(h[:, 0].float(), dim=-1).cpu()
            out[f"{tag}_pat"] = F.normalize(h[:, 1:].float().mean(1), dim=-1).cpu()
        return out


def cat_batches(chunks):
    return {k: torch.cat([c[k] for c in chunks], 0) for k in chunks[0]}


def dump_compbench(args, device, dino):
    prompts = load_prompts()
    p2i = prompt_to_idx()
    od = Path(args.out); od.mkdir(parents=True, exist_ok=True)
    if (od / "compbench_img.pt").exists():     # resume: 19k DINO encodes is the expensive half
        per_cand = torch.load(od / "compbench_img.pt", map_location="cpu")
        keys = sorted(set.intersection(*[set(per_cand[c]["idx"].tolist()) for c in range(8)]))
        print(f"[cb] reusing existing compbench_img.pt ({len(keys)} prompts), text stage only", flush=True)
        txt = encode_text(args, device, [prompts[i] for i in keys]); txt["idx"] = torch.tensor(keys)
        torch.save(txt, od / "compbench_txt.pt")
        print(f"[cb] wrote {od}/compbench_txt.pt")
        return
    per_cand = {}
    for c in range(8):
        paths, idxs, miss = [], [], 0
        for f in sorted(glob.glob(f"phaseX/candeval/cand{c}/cb_*/samples/*.png")):
            m = re.match(r"^(.*)_(\d{6})$", Path(f).stem)
            if m is None:
                raise SystemExit(f"cannot parse filename {f}")
            key = (Path(f).parts[-3][3:], m.group(1).strip())     # cb_3d_spatial -> 3d_spatial
            if key not in p2i:
                miss += 1; continue
            paths.append(f); idxs.append(p2i[key])
        if len(set(idxs)) != len(idxs):
            raise SystemExit(f"cand{c}: repeated prompt index after the text join")
        if miss:
            print(f"[cb] cand{c}: {miss} samples had no labelled prompt, skipped", flush=True)
        chunks = []
        for s in range(0, len(paths), args.batch):
            ims = [Image.open(p).convert("RGB") for p in paths[s:s + args.batch]]
            chunks.append(dino(ims))
            if (s // args.batch) % 40 == 0:
                print(f"[cb] cand{c} {s}/{len(paths)}", flush=True)
        d = cat_batches(chunks); d["idx"] = torch.tensor(idxs)
        per_cand[c] = d
        # Independent check on the join against phaseX/cand_dino, which was built by a different
        # script with its own image->index mapping. That script center-crops and bicubic-resizes by
        # hand where this one uses the HF AutoImageProcessor, so the two CLS vectors are NOT equal
        # (~0.95 cosine) and an equality test would be testing preprocessing, not the join. The
        # DISCRIMINATIVE test is what matters: my feature for prompt i must be closer to the cached
        # feature for prompt i than to the cached feature for some other prompt. A scrambled join
        # fails this immediately; a preprocessing difference does not.
        old = torch.load(f"phaseX/cand_dino/compbench_cand{c}.pt", map_location="cpu")
        opos = {int(i): k for k, i in enumerate(old["idx"].tolist())}
        mine = F.normalize(d["l_cls"].float(), dim=-1)
        ocls = F.normalize(old["cls"].float(), dim=-1)
        same = (mine * ocls[[opos[i] for i in idxs]]).sum(-1)
        g = torch.Generator().manual_seed(0)
        shuf = torch.randperm(len(idxs), generator=g)
        other = (mine * ocls[[opos[idxs[j]] for j in shuf.tolist()]]).sum(-1)
        win = float((same > other).float().mean())
        print(f"[cb] cand{c}: {len(idxs)} images | join check: same-prompt cos {same.mean():.4f} "
              f"vs other-prompt {other.mean():.4f}, closer on {100*win:.2f}% of images", flush=True)
        if win < 0.98:
            raise SystemExit(f"cand{c}: join looks scrambled (same-prompt closer on only {100*win:.1f}%)")

    od = Path(args.out); od.mkdir(parents=True, exist_ok=True)
    torch.save(per_cand, od / "compbench_img.pt")

    keys = sorted(set.intersection(*[set(per_cand[c]["idx"].tolist()) for c in range(8)]))
    txt = encode_text(args, device, [prompts[i] for i in keys])
    txt["idx"] = torch.tensor(keys)
    torch.save(txt, od / "compbench_txt.pt")
    print(f"[cb] wrote {od}/compbench_img.pt and compbench_txt.pt  ({len(keys)} shared prompts)")


def dump_coco(args, device, dino):
    from diffusers import StableDiffusion3Pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).to(device)
    pipe.vae.to(dtype=torch.bfloat16).eval(); pipe.set_progress_bar_config(disable=True)

    R = {}
    for f in sorted(glob.glob("phaseN/coco_selection_108348/selection_rank*.jsonl")):
        for line in open(f):
            r = json.loads(line); R[r["idx"]] = r

    od = Path(args.out)
    if (od / "coco_img.pt").exists():          # resume: the decode is the expensive half
        idx = torch.load(od / "coco_img.pt", map_location="cpu")["idx"].tolist()
        print(f"[coco] reusing existing coco_img.pt ({len(idx)} captions), text stage only", flush=True)
        txt = encode_text(args, device, [R[int(i)]["prompt"] for i in idx], pipe=pipe)
        txt["idx"] = torch.tensor(idx)
        torch.save(txt, od / "coco_txt.pt")
        print(f"[coco] wrote {od}/coco_txt.pt")
        return
    data = [torch.load(f, map_location="cpu") for f in sorted(glob.glob("phaseT/energy_cache/latents_shard*.pt"))]
    idx = [i for d in data for i in d["idx"]]
    cand = torch.cat([d["cand"] for d in data], 0)
    N, P = int(data[0]["N"]), len(idx)
    print(f"[coco] {P} captions x N={N}", flush=True)

    @torch.no_grad()
    def decode(z):
        img = pipe.vae.decode(z.to(device, torch.bfloat16) / pipe.vae.config.scaling_factor
                              + pipe.vae.config.shift_factor).sample
        img = ((img.float().clamp(-1, 1) + 1) / 2 * 255).round().to(torch.uint8)
        return [Image.fromarray(x.permute(1, 2, 0).cpu().numpy()) for x in img]

    cand_chunks, ref_chunks = [], []
    for k in range(P):
        rec = R[idx[k]]
        ref = Path(str(rec["reference"]))
        if not ref.suffix:
            ref = ref.with_suffix(".jpg")
        ref_img = Image.open(ref).convert("RGB").resize((args.height, args.height), Image.BICUBIC)
        cand_chunks.append(dino(decode(cand[k])))
        ref_chunks.append(dino([ref_img]))
        if (k + 1) % 300 == 0:
            print(f"[coco] {k+1}/{P}", flush=True)

    d = {f"cand_{k}": torch.stack([c[k] for c in cand_chunks], 0) for k in cand_chunks[0]}   # [P,N,D]
    d.update({f"ref_{k}": torch.cat([c[k] for c in ref_chunks], 0) for k in ref_chunks[0]})  # [P,D]
    d["idx"] = torch.tensor(list(idx)); d["N"] = N
    od = Path(args.out); od.mkdir(parents=True, exist_ok=True)
    torch.save(d, od / "coco_img.pt")

    txt = encode_text(args, device, [R[i]["prompt"] for i in idx], pipe=pipe)
    txt["idx"] = torch.tensor(list(idx))
    torch.save(txt, od / "coco_txt.pt")
    print(f"[coco] wrote {od}/coco_img.pt and coco_txt.pt")


@torch.no_grad()
def encode_text(args, device, texts, pipe=None):
    if pipe is None:
        from diffusers import StableDiffusion3Pipeline
        pipe = StableDiffusion3Pipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).to(device)
        pipe.set_progress_bar_config(disable=True)
    # `torch_dtype=bfloat16` does not reach CLIP's `text_projection`, which the SD3.5 repo ships in
    # fp16; the mismatch only surfaces inside _get_clip_prompt_embeds. Cast the encoders explicitly.
    for te in (pipe.text_encoder, getattr(pipe, "text_encoder_2", None), getattr(pipe, "text_encoder_3", None)):
        if te is not None:
            te.to(device=device, dtype=torch.bfloat16)
    pooled, seqmean = [], []
    for s in range(0, len(texts), 16):
        pe, _, pp, _ = pipe.encode_prompt(prompt=texts[s:s + 16], prompt_2=None, prompt_3=None,
                                          do_classifier_free_guidance=False, device=device)
        pooled.append(pp.float().cpu()); seqmean.append(pe.float().mean(1).cpu())
        if (s // 16) % 40 == 0:
            print(f"[txt] {s}/{len(texts)}", flush=True)
    return {"pooled": torch.cat(pooled, 0), "seqmean": torch.cat(seqmean, 0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["compbench", "coco"], required=True)
    ap.add_argument("--out", default="phaseW/feats")
    ap.add_argument("--model_id", default="stabilityai/stable-diffusion-3.5-medium")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--height", type=int, default=512)
    args = ap.parse_args()
    device = torch.device("cuda")
    dino = Dino(device)
    (dump_compbench if args.set == "compbench" else dump_coco)(args, device, dino)


if __name__ == "__main__":
    main()
