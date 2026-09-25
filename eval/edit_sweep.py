#!/usr/bin/env python3
"""FlowEdit sweep for the SoftREPA-style editing tables: one model, one dataset, many settings (docs/editing/).

Reuses the reviewed FlowEdit implementation unchanged: eval/flowedit.py (the update, VAE encode, scheduler grid,
pipeline loading), eval/editing/noise.py (SHA-256 paired noise), common/sampling.py (prompt encoding, VAE decode).
Nothing here re-implements the update. The additions are the dataset adapters and a loop that encodes each record
once and runs every requested setting on it.

Pairing: the noise at interval i of a T-step grid is a pure function of (seed, record id, i), and the VAE posterior
draw of (seed, record id), so every model sees identical noise for the same (record, T).

SoftREPA consistency (their sampler.py, SD3EulerFE):
  * scheduler grid of T steps (SD3 flow-match scheduler, shift 3.0), the last n intervals active;
  * VAE posterior SAMPLE (seeded), not mode;
  * source CFG in the standard form v_u + s (v_c - v_u);
  * SoftREPA's TARGET CFG is v_c + w (v_c - v_u) = standard form with s = w + 1. Settings take the STANDARD s
    ('tgt'); the tables print SoftREPA's w = s - 1.

Setting string "T:n:src:tgt" (standard guidance), separated by '+' (bsub -env splits on commas) or ','.
Students are K=8 models: run them on T=8, whose sigmas are exactly their training grid.

  python eval/edit_sweep.py --dataset div2k --model ours118k --settings 8:5:1:3+8:7:1:2
Data: PIEBENCH_ROOT (PIE-Bench_v1 with mapping_file.json and annotation_images/), EDIT_DATA (holds div2k_set/ and
cat2dog_set/ from data/build_edit_prompts.py), CKPT_ROOT (the checkpoints/ tree of docs/CHECKPOINTS.md).
"""
from __future__ import annotations

import argparse, contextlib, io, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from common.sampling import encode_prompt, vae_decode  # noqa: E402
from eval.editing.noise import build_noise_bank, vae_seed  # noqa: E402
from eval.flowedit import encode_source_latent, flowedit, load_pipeline, scheduler_grid  # noqa: E402

PIE_ROOT = Path(os.environ.get("PIEBENCH_ROOT", "/lustre/scratch126/cellgen/lotfollahi/ap55/EnergyVLM/PIE-Bench_v1"))
EDIT_DATA = Path(os.environ.get("EDIT_DATA", REPO / "data/editing"))
CK = Path(os.environ.get("CKPT_ROOT", "/lustre/scratch126/cellgen/lotfollahi/ha11/EnergyVLM/checkpoints"))
MODELS = {
    "teacher": None,                                                        # stock SD3.5-M transformer
    "ours118k": CK / "phaseW/phaseW_CD_dinop_hard_118k-rewX_s0_138926/checkpoint_avg_last5.pt",
    "naive118k": CK / "phaseW/phaseW_B2_118k_s0_128712/checkpoint_avg_last5.pt",
    "naiveS4": CK / "phaseS4/phaseS4_B2_s1_130451/checkpoint_final.pt",
}
N_RECORDS = {"pie": 700, "div2k": 800, "cat2dog": 500}


def clean_prompt(text: str) -> str:
    """PIE-Bench's [ ] edit markers dropped, as the official evaluator does (and strip)."""
    return text.replace("[", "").replace("]", "").strip()


def load_dataset(name: str) -> list[dict]:
    """[{id, image (abs path, 512x512), src, tgt}] in a fixed order (PIE: sorted record ids)."""
    if name == "pie":
        mapping = json.loads((PIE_ROOT / "mapping_file.json").read_text())
        recs = [{"id": rid, "image": str(PIE_ROOT / "annotation_images" / mapping[rid]["image_path"]),
                 "src": clean_prompt(mapping[rid]["original_prompt"]),
                 "tgt": clean_prompt(mapping[rid]["editing_prompt"]),
                 "category_id": int(mapping[rid]["editing_type_id"])} for rid in sorted(mapping)]
    else:
        d = EDIT_DATA / f"{name}_set"
        recs = [{"id": r["id"], "image": str(d / r["image"]), "src": r["source_prompt"], "tgt": r["target_prompt"]}
                for r in json.loads((d / "records.json").read_text())]
    assert len(recs) == N_RECORDS[name], f"{name}: {len(recs)} records, expected {N_RECORDS[name]}"
    return recs


def load_image(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    assert img.size == (512, 512), f"{path} is {img.size}"
    x = torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).permute(2, 0, 1).float().div_(255.0)
    return (x * 2.0 - 1.0).unsqueeze(0)


def load_student(pipe, path: Path, device) -> dict:
    """Every transformer key must match (same contract as eval/editing/model_registry.load_transformer)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    missing, unexpected = pipe.transformer.load_state_dict(ck["model"], strict=False)
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    pipe.transformer.to(device=device, dtype=torch.bfloat16).eval()
    prov = {"checkpoint": str(path), "checkpoint_step": ck.get("step"), "n_tensors": len(ck["model"])}
    del ck
    return prov


def parse_settings(text: str) -> list[dict]:
    out = []
    for s in text.replace("+", ",").split(","):
        T, n, src, tgt = s.split(":")
        T, n = int(T), int(n)
        assert 1 <= n <= T
        out.append({"T": T, "n": n, "src": float(src), "tgt": float(tgt),
                    "tag": f"T{T}_n{n}_s{float(src):g}_t{float(tgt):g}"})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(N_RECORDS), required=True)
    ap.add_argument("--model", required=True, help=f"one of {sorted(MODELS)}, or any label with --checkpoint")
    ap.add_argument("--checkpoint", default=None, help="student checkpoint path ('base' = the stock transformer)")
    ap.add_argument("--settings", required=True)
    ap.add_argument("--out", type=Path, default=REPO / "out/editing")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", default="0/1", help="i/N: this job takes records i, i+N, ...")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reverse", action="store_true",
                    help="walk this shard from the end: a helper for a running shard; the two meet in the middle and "
                         "the per-image resume check skips what the other already wrote")
    a = ap.parse_args()
    settings = parse_settings(a.settings)
    si, sn = map(int, a.shard.split("/"))
    recs = load_dataset(a.dataset)
    if a.limit:
        recs = recs[:a.limit]
    mine = recs[si::sn]
    if a.reverse:
        mine = mine[::-1]
    ckpt = a.checkpoint if a.checkpoint is not None else MODELS[a.model]
    device = torch.device("cuda")
    pipe = load_pipeline("stabilityai/stable-diffusion-3.5-medium", device)
    prov = {"checkpoint": "base"} if ckpt in (None, "base") else load_student(pipe, Path(ckpt), device)
    grids = {}
    for T in sorted({s["T"] for s in settings}):
        sig, ts = scheduler_grid(pipe.scheduler, T, device)
        grids[T] = (sig, ts)
        print(f"[sweep] T={T} sigmas {[round(float(x), 5) for x in sig]}", flush=True)
    base = a.out / a.dataset / a.model
    for s in settings:
        (base / s["tag"]).mkdir(parents=True, exist_ok=True)
    meta_path = base / f"meta_shard{si}of{sn}{'_rev' if a.reverse else ''}.jsonl"
    print(f"[sweep] {a.dataset} {a.model} {prov} | {len(mine)} records (shard {a.shard}) x {len(settings)} settings",
          flush=True)
    t0 = time.time()
    with open(meta_path, "a") as meta_f:
        for k, r in enumerate(mine):
            todo = [s for s in settings if not (base / s["tag"] / f"{r['id']}.png").is_file()]
            if not todo:
                continue
            with torch.no_grad():
                src_img = load_image(r["image"])
                z_src = encode_source_latent(pipe, src_img, device, a.seed, mode="sample",
                                             explicit_seed=vae_seed(a.seed, r["id"]))
                se, spo = encode_prompt(pipe, r["src"], device)
                te, tpo = encode_prompt(pipe, r["tgt"], device)
                ne, npo = encode_prompt(pipe, "", device)
                banks = {T: build_noise_bank(T, 1, tuple(z_src.shape), device, a.seed, r["id"])
                         for T in {s["T"] for s in todo}}
                for s in todo:
                    sig, ts = grids[s["T"]]
                    active = list(range(s["T"] - s["n"], s["T"]))
                    bank, seeds = banks[s["T"]]
                    with contextlib.redirect_stdout(io.StringIO()):       # flowedit prints per interval
                        z, counts = flowedit(pipe, z_src, sig, ts, active, bank, se, spo, te, tpo, ne, npo,
                                             s["src"], s["tgt"], device)
                    img = ((vae_decode(pipe.vae, z) + 1.0) / 2.0).clamp(0.0, 1.0)[0]
                    arr = img.mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
                    p = base / s["tag"] / f"{r['id']}.png"
                    tmp = p.with_name(p.stem + f".tmp{os.getpid()}.png")
                    Image.fromarray(arr).save(tmp)
                    os.replace(tmp, p)
                    meta_f.write(json.dumps({"id": r["id"], "setting": s["tag"], "active": active,
                                             "sigmas": [round(float(sig[i]), 6) for i in active],
                                             "forwards": counts, "noise_seed0": seeds[active[0]][0],
                                             **prov}) + "\n")
                meta_f.flush()
            if k % 20 == 0:
                print(f"[sweep] {k + 1}/{len(mine)} records, {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"[sweep] DONE {a.dataset} {a.model} shard {a.shard} in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
