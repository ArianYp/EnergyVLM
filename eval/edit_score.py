#!/usr/bin/env python3
"""Score FlowEdit outputs with SoftREPA's metrics (all datasets) and the official PIE-Bench metrics (PIE only).

SoftREPA block: literal ports of SoftREPA's eval.py + eval_utils.py, scored against the TARGET prompt.
  image_reward  ImageReward-v1.0 .score(prompt, [path])                          (raw; x100 in Table 2 layouts)
  pickscore     yuvalkirstain/PickScore_v1 + laion CLIP-ViT-H-14 processor, logit_scale.exp() * cos
  hps           hpsv2.score([path], prompt, hps_version="v2.1")
  clip          ImageReward's CLIP scorer: load_score("CLIP").inference_rank(prompt, [path]) = cos, ViT-L/14
  lpips_sr      transforms.Resize((299, 299)) -> ToTensor() -> (x*255).byte(), lpips.LPIPS(net='vgg')(edited, source)
                (0..255 inputs and PIL's antialiased resize, exactly as SoftREPA's eval.py)

PIE-Bench block: the UNMODIFIED official evaluator (cure-lab/PnPInversion evaluation/, vendored in eval/official_pie/
with SOURCE.txt and SHA256SUMS). Covers structure_distance, psnr/lpips/mse/ssim on the whole image and on the
unedited part (NaN where the edit mask covers the whole image, as the official code returns),
clip_similarity_target_image and clip_similarity_target_image_edit_part. Masks come from the official mask_decode.

Output: <out>/<dataset>/<model>/scores_<setting>.jsonl, one line per record; resumable per record.
  python eval/edit_score.py --dataset pie --model ours118k [--settings T8_n5_s1_t3+T8_n7_s1_t2]
ImageReward's two scorers read their weights from IMAGEREWARD_ROOT (default ~/.cache/ImageReward: ImageReward.pt,
med_config.json, and CLIP's ViT-L-14.pt with sha256 b8cca3fd...a03836).
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eval/official_pie"))
from eval.edit_sweep import PIE_ROOT, load_dataset  # noqa: E402  (same records, order and prompts as generation)

IR_ROOT = Path(os.environ.get("IMAGEREWARD_ROOT", Path.home() / ".cache/ImageReward"))
PIE_METRICS = ["structure_distance", "psnr", "lpips", "mse", "ssim",
               "psnr_unedit_part", "lpips_unedit_part", "mse_unedit_part", "ssim_unedit_part",
               "clip_similarity_target_image", "clip_similarity_target_image_edit_part"]


class SoftREPAMetrics:
    def __init__(self, device="cuda"):
        from torchvision import transforms
        import lpips
        # ImageReward's package __init__ imports pandas, which breaks against NumPy 2 in some envs; the shim imports
        # the scorer modules directly. Same models and code paths as RM.load / RM.load_score.
        from eval.editing.metrics.imagereward_compat import load_imagereward
        self.ir = load_imagereward("ImageReward-v1.0", device=device, download_root=IR_ROOT)
        import ImageReward.utils as RMU
        self.clip = RMU.load_score("CLIP", device=device, download_root=str(IR_ROOT))
        from transformers import AutoModel, AutoProcessor
        self.pick_proc = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.pick = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").eval().to(device)
        import hpsv2
        self.hpsv2 = hpsv2
        self.lp = lpips.LPIPS(net="vgg").to(device)
        self.tf = transforms.Compose([transforms.Resize((299, 299)), transforms.ToTensor(),
                                      lambda x: (x * 255).byte()])
        self.device = device

    @torch.no_grad()
    def pickscore(self, prompt, path):
        img = self.pick_proc(images=[Image.open(path)], padding=True, truncation=True, max_length=77,
                             return_tensors="pt").to(self.device)
        txt = self.pick_proc(text=prompt, padding=True, truncation=True, max_length=77,
                             return_tensors="pt").to(self.device)
        ie = self.pick.get_image_features(**img); ie = ie / torch.norm(ie, dim=-1, keepdim=True)
        te = self.pick.get_text_features(**txt); te = te / torch.norm(te, dim=-1, keepdim=True)
        return float((self.pick.logit_scale.exp() * (te @ ie.T)[0])[0])

    @torch.no_grad()
    def __call__(self, prompt, edited_path, source_path) -> dict:
        ir = self.ir.score(prompt, [str(edited_path)])
        _, clip = self.clip.inference_rank(prompt, [str(edited_path)])
        hps = self.hpsv2.score([str(edited_path)], prompt, hps_version="v2.1")
        a = self.tf(Image.open(edited_path).convert("RGB")).unsqueeze(0).to(self.device)
        b = self.tf(Image.open(source_path).convert("RGB")).unsqueeze(0).to(self.device)
        f = lambda v: float(v[0] if isinstance(v, (list, tuple)) else v)
        return {"image_reward": f(ir), "pickscore": self.pickscore(prompt, edited_path), "hps": f(hps),
                "clip": f(clip), "lpips_sr": float(self.lp(a, b).item())}


class PieOfficial:
    def __init__(self, device="cuda"):
        from official_evaluate import calculate_metric, mask_decode
        from evaluation.matrics_calculator import MetricsCalculator
        self.calc = MetricsCalculator(device)
        self.calculate_metric, self.mask_decode = calculate_metric, mask_decode
        self.map = json.loads((PIE_ROOT / "mapping_file.json").read_text())

    def __call__(self, rid, edited_path, source_path) -> dict:
        item = self.map[rid]
        mask = self.mask_decode(item["mask"])[:, :, np.newaxis].repeat([3], axis=2)
        src_p = item["original_prompt"].replace("[", "").replace("]", "")
        tgt_p = item["editing_prompt"].replace("[", "").replace("]", "")
        src, tgt = Image.open(source_path), Image.open(edited_path)
        out = {}
        for m in PIE_METRICS:
            v = self.calculate_metric(self.calc, m, src, tgt, mask, mask, src_p, tgt_p)
            out[f"pie_{m}"] = float(np.asarray(v).reshape(-1)[0]) if v is not None and v == v else float("nan")
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["pie", "div2k", "cat2dog"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--settings", default="all", help="setting tags separated by '+' or ',', or 'all'")
    ap.add_argument("--out", type=Path, default=REPO / "out/editing")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    base = a.out / a.dataset / a.model
    tags = (sorted(p.name for p in base.iterdir() if p.is_dir()) if a.settings == "all"
            else a.settings.replace("+", ",").split(","))
    recs = load_dataset(a.dataset)
    if a.limit:
        recs = recs[:a.limit]
    sr = SoftREPAMetrics()
    pie = PieOfficial() if a.dataset == "pie" else None
    t0 = time.time()
    for tag in tags:
        outp = base / f"scores_{tag}.jsonl"
        done = {json.loads(l)["id"] for l in outp.read_text().splitlines()} if outp.exists() else set()
        todo = [r for r in recs if r["id"] not in done]
        missing = [r["id"] for r in todo if not (base / tag / f"{r['id']}.png").is_file()]
        if missing:
            print(f"[score] {tag}: {len(missing)} images missing (e.g. {missing[:2]}); scoring the rest", flush=True)
        with open(outp, "a") as f:
            for r in todo:
                p = base / tag / f"{r['id']}.png"
                if not p.is_file():
                    continue
                row = {"id": r["id"], **sr(r["tgt"], p, r["image"])}
                if pie is not None:
                    row.update(pie(r["id"], p, r["image"]))
                f.write(json.dumps(row) + "\n")
        n = len(outp.read_text().splitlines())
        print(f"[score] {a.dataset}/{a.model}/{tag}: {n}/{len(recs)} scored, {(time.time() - t0) / 60:.1f} min",
              flush=True)
    print("[score] DONE", flush=True)


if __name__ == "__main__":
    main()
