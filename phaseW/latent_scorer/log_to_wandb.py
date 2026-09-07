#!/usr/bin/env python3
"""Push the latent-scorer study's on-disk records to Weights & Biases after the fact: the projector
training curves and test metrics (with the checkpoint as an artifact), the gradient diagnostic at
both model states, and the seed-level selection-rule statistics and fidelity tables.

    python3 phaseW/latent_scorer/log_to_wandb.py --project sd-phaseW-s4
"""
import argparse, json, re
from pathlib import Path
import wandb

ap = argparse.ArgumentParser(); ap.add_argument("--project", default="sd-phaseW-s4"); args = ap.parse_args()

# 1. projector training
pdir = Path("phaseW/latent_scorer/projector")
log = json.load(open(pdir / "train_log.json")); test = json.load(open(pdir / "test_metrics.json"))
import torch
cfg = torch.load(pdir / "projector.pt", map_location="cpu", weights_only=False).get("args", {})
run = wandb.init(project=args.project, name="latent-projector", job_type="latent-scorer",
                 config={**cfg, "train_captions": 22000, "val_captions": 2000, "test_captions": 3000, "params_M": 6.8,
                         "architecture": "5-stage conv (64,128,256,512,512), GN+SiLU, global mean pool, MLP 512-1024-768, L2-normalised",
                         "loss": "mean(1-cos(P(z), DINO(candidate))) + lambda*KL(softmax(S_rgb/tau) || softmax(S_hat/tau))"})
for m in log["log"]:
    run.log({f"proj/{k}": v for k, v in m.items() if k != "epoch"}, step=m["epoch"])
run.summary.update({f"test/{k}": v for k, v in test["test"].items()})
run.summary.update({f"test_vs_cached/{k}": v for k, v in test["test_vs_cached"].items()})
run.summary.update({f"best_val/{k}": v for k, v in test["best_val"].items()})
art = wandb.Artifact("latent-projector", type="model", metadata={"best_epoch": test["best_val"]["epoch"], **{f"test_{k}": v for k, v in test["test"].items()}})
for f in ("projector.pt", "train_log.json", "test_metrics.json"):
    art.add_file(str(pdir / f))
run.log_artifact(art); run.finish()
print("logged latent-projector")

# 2. gradient diagnostic
for name, f in (("grad-diag-argmax-student-s0", "phaseW/grad_diag_hard_s0.json"), ("grad-diag-pretrained-init", "phaseW/grad_diag_base.json")):
    d = json.load(open(f))
    run = wandb.init(project=args.project, name=name, job_type="gradient-diagnostic", config={"ckpt": d["ckpt"], "temp": d["temp"], "grad_clip": d["grad_clip"], "n_params": d["n_params"], "n_captions": len(d["captions"])})
    cols = ["idx", "prompt", "argmax", "ess_entropy", "ess_kish", "p_switch", "gF_norm", "V_c_rel", "adam_ratio", "B_c_rel", "cos_Eclip_vs_clipF", "Eclip_norm_rel", "cos_argmax_F", "cos_pairwise_mean", "delta_ref_cos", "delta_ref_rel_sq_diff", "batch_cos_grad"]
    tab = wandb.Table(columns=cols)
    for c in d["captions"]:
        tab.add_data(c["idx"], c["prompt"], c["argmax"], c["ess_entropy"], c["ess_kish"], c["p_switch"], c["gF_norm"], c["V_c_rel"], c["adam_ratio"], c["B_c_rel"], c["cos_Eclip_vs_clipF"], c["Eclip_norm_rel"], c["cos_argmax_F"], c["cos_pairwise_mean"], c["delta_ref"]["cos"], c["delta_ref"]["rel_sq_diff"], c["batch_shape"]["cos_grad"])
    run.log({"diag/per_caption": tab})
    for k, v in d["summary"].items():
        if isinstance(v, dict):
            run.summary[f"diag/{k}_mean"] = v["mean"]; run.summary[f"diag/{k}_median"] = v["median"]
        else:
            run.summary[f"diag/{k}"] = v
    run.finish(); print("logged", name)

# 3. seed statistics + fidelity tables
run = wandb.init(project=args.project, name="selection-rule-seed-stats", job_type="analysis")
for suffix in ("-raw", "-avglast3"):
    d = json.load(open(f"phaseW/s4_seed_stats{suffix}.json"))
    tab = wandb.Table(columns=["contrast", "cb_per_seed", "cb_mean", "cb_sd", "cb_ci_lo", "cb_ci_hi", "cb_p", "ge_per_seed", "ge_mean", "ge_sd", "ge_p"])
    for k, v in d.items():
        c, g = v["compbench"], v["geneval2"]
        tab.add_data(k, str([round(x, 4) for x in c["per_seed"]]), c["mean"], c["sd"], c["ci95"][0], c["ci95"][1], c["t_p"], str([round(x, 2) for x in g["per_seed"]]), g["mean"], g["sd"], g["t_p"])
    run.log({f"seed_stats/contrasts{suffix}": tab})
md = open("phaseW/s4_seed_stats.md").read()
run.log({"seed_stats/markdown": wandb.Html("<pre>" + md.replace("<", "&lt;") + "</pre>")})
fid = Path("phaseW/fidelity_s4_report.md")
if fid.is_file():
    tab = wandb.Table(columns=["model", "FID", "CMMD", "precision", "recall"])
    for ln in fid.read_text().splitlines():
        m = re.match(r"\|\s*(S4_\S+)@4\s*\|\s*\d+\s*\|\s*([\d.]+)\s*\|[^|]*\|\s*([\d.]+)\s*\|[^|]*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|", ln)
        if m: tab.add_data(m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5)))
    run.log({"fidelity/selection_rule_arms": tab})
run.finish(); print("logged selection-rule-seed-stats")
