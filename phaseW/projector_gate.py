#!/usr/bin/env python3
"""The projector gate declared in phaseW/PREREGISTRATION_projector_gate.md.

Fits the proposed learned DINO reward -- R = cos(g(f(x)), g(f_bar)) with a learnable projector g --
on cached candidates and asks whether it out-ranks the fixed cosine it is meant to replace.

Everything is selected on a VAL split of prompts and reported on a TEST split of prompts; the two
are prompt-disjoint from train, so a projector that memorises prompt identity gains nothing. That
disjointness is the whole point: an image-only readout already predicts the CompBench label at
pooled r = +0.394 by reading WHICH prompt an image came from, and 77.6% of the label variance is
between-prompt. Only the within-prompt component is actionable for selection, so within-prompt
Spearman is the selection criterion and the reported endpoint.

Arms, per the preregistration:
    cos_fixed   the incumbent fixed cosine                            no learning, no text
    probe       ridge on the candidate feature alone                  no anchor,   no text
    proj_A1     cos(g u, g v), identity-anchored     g = I + dW       the proposal's A1
    proj_A2     cos(g u, g v), reconstruction-anchored via h(g(u))    the proposal's A2
    text_head   two-tower cos(P_i u, P_t e) with an interaction term  NOT in the proposal

On CompBench the anchor is the highest-LABELLED sibling -- an oracle the method would never have --
and every arm is scored on the same 7 non-anchor candidates so the comparison is matched. On COCO
the anchor is the genuine reference photograph, which is the proposal's actual setting.
"""
from __future__ import annotations

import argparse, collections, glob, json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr, wilcoxon

REPS = ["l_cls", "l_pat", "b_cls", "b_pat"]


# --------------------------------------------------------------------------------------- data ----
def load_compbench(root: Path):
    lab = collections.defaultdict(dict); cat = {}
    for f in sorted(glob.glob("phaseX/candeval/cand*/candidate_labels.json")):
        r = json.load(open(f)); c = r["candidate"]
        for k, d in r["compbench"].items():
            for pp in d["per_prompt"]:
                lab[pp["idx"]][c] = pp["score"]; cat[pp["idx"]] = k
    vq = collections.defaultdict(dict)
    for f in glob.glob("phaseN/bestofk_rec_113031/compbench_vqa_rank*.jsonl"):
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                for j, v in enumerate(r["vqa"]): vq[int(r["idx"])][j] = v

    img = torch.load(root / "compbench_img.pt", map_location="cpu")
    txt = torch.load(root / "compbench_txt.pt", map_location="cpu")
    pos = [{int(i): k for k, i in enumerate(img[c]["idx"].tolist())} for c in range(8)]
    tpos = {int(i): k for k, i in enumerate(txt["idx"].tolist())}
    keys = sorted(i for i in lab
                  if len(lab[i]) == 8 and len(vq.get(i, {})) == 8 and i in tpos
                  and all(i in p for p in pos))
    Y = np.array([[lab[i][c] for c in range(8)] for i in keys], dtype=np.float64)
    V = np.array([[vq[i][c] for c in range(8)] for i in keys], dtype=np.float64)
    X = {r: torch.stack([torch.stack([img[c][r][pos[c][i]] for c in range(8)]) for i in keys]).float()
         for r in REPS}
    T = torch.cat([txt["pooled"], txt["seqmean"]], -1)[[tpos[i] for i in keys]].float()
    anchor = Y.argmax(1)                                   # ORACLE anchor: best-labelled sibling
    return dict(Y=Y, V=V, X=X, T=T, anchor=anchor, cat=np.array([cat[i] for i in keys]), keys=keys)


def load_coco(root: Path):
    R = {}
    for f in sorted(glob.glob("phaseN/coco_selection_108348/selection_rank*.jsonl")):
        for line in open(f):
            r = json.loads(line); R[r["idx"]] = r
    img = torch.load(root / "coco_img.pt", map_location="cpu")
    txt = torch.load(root / "coco_txt.pt", map_location="cpu")
    keys = [int(i) for i in img["idx"].tolist()]
    Y = np.array([R[i]["endpoint_vqa"] for i in keys], dtype=np.float64)
    X = {r: img[f"cand_{r}"].float() for r in REPS}
    A = {r: img[f"ref_{r}"].float() for r in REPS}          # REAL photograph anchor
    T = torch.cat([txt["pooled"], txt["seqmean"]], -1).float()
    return dict(Y=Y, X=X, A=A, T=T, keys=keys)


# ------------------------------------------------------------------------------------- scoring ----
def within_rho(S, Y, mask=None):
    """Per-prompt Spearman between a score matrix and the label matrix."""
    out = []
    for k in range(len(Y)):
        y, s = Y[k], S[k]
        if mask is not None:
            y, s = y[mask[k]], s[mask[k]]
        if np.std(y) < 1e-12 or np.std(s) < 1e-12:
            out.append(np.nan); continue
        out.append(spearmanr(s, y)[0])
    return np.array(out)


def headroom(S, Y):
    rng = np.random.default_rng(0)
    rand = np.mean([Y[k, rng.integers(Y.shape[1])] for k in range(len(Y))])
    return 100.0 * (np.mean([Y[k, S[k].argmax()] for k in range(len(Y))]) - rand) / (Y.max(1).mean() - rand)


# -------------------------------------------------------------------------------------- models ----
class ProjA1(nn.Module):
    """g(u) = u + dW u, so mu * E||g(u)-u||^2 = mu * E||dW u||^2 anchors g to the identity."""
    def __init__(self, d, **_):
        super().__init__(); self.dW = nn.Linear(d, d, bias=False); nn.init.zeros_(self.dW.weight)

    def g(self, u): return u + self.dW(u)

    def score(self, U, a):
        return F.cosine_similarity(self.g(U), self.g(a).unsqueeze(1), dim=-1)

    def omega(self, U): return self.dW(U).pow(2).sum(-1).mean()


class ProjA2(nn.Module):
    """g: d -> k free, anchored by requiring h(g(u)) to reconstruct u."""
    def __init__(self, d, k=128, **_):
        super().__init__(); self.g_ = nn.Linear(d, k, bias=False); self.h = nn.Linear(k, d, bias=False)

    def g(self, u): return self.g_(u)

    def score(self, U, a):
        return F.cosine_similarity(self.g(U), self.g(a).unsqueeze(1), dim=-1)

    def omega(self, U): return (self.h(self.g(U)) - U).pow(2).sum(-1).mean()


class TextHead(nn.Module):
    """Two-tower with an explicit image x text interaction; the anchor is concatenated when present."""
    def __init__(self, d, dt=0, k=256, use_anchor=True, **_):
        super().__init__()
        din = d * (2 if use_anchor else 1)
        self.use_anchor = use_anchor
        self.pi = nn.Linear(din, k); self.pt = nn.Linear(dt, k)
        self.head = nn.Sequential(nn.Linear(3 * k, k), nn.GELU(), nn.Linear(k, 1))

    def score(self, U, a, e=None):
        z = torch.cat([U, a.unsqueeze(1).expand_as(U)], -1) if self.use_anchor else U
        zi = self.pi(z)                                        # [B,N,k]
        zt = self.pt(e).unsqueeze(1).expand_as(zi)             # [B,N,k]
        return self.head(torch.cat([zi, zt, zi * zt], -1)).squeeze(-1)

    def omega(self, U): return torch.zeros((), device=U.device)


MODELS = {"proj_A1": ProjA1, "proj_A2": ProjA2, "text_head": TextHead}


# ---------------------------------------------------------------------------------------- loss ----
def rank_loss(s, y, kind, kappa):
    """s,y: [B,N]. Pairwise-logistic (tie-robust) or Plackett-Luce on the label order."""
    if kind == "pair":
        di = s.unsqueeze(2) - s.unsqueeze(1)
        dy = y.unsqueeze(2) - y.unsqueeze(1)
        w = (dy.abs() > 1e-8).float() * dy.sign()
        n = w.abs().sum().clamp(min=1.0)
        return (F.softplus(-w * di / kappa) * w.abs()).sum() / n
    order = torch.argsort(y, dim=1, descending=True)
    so = torch.gather(s, 1, order) / kappa
    lse = torch.flip(torch.logcumsumexp(torch.flip(so, [1]), dim=1), [1])
    return (lse - so).mean()


def fit(mk, Xtr, Atr, Ytr, Ttr, val, cfg, device, seed=0):
    torch.manual_seed(seed)
    d, dt = Xtr.shape[-1], Ttr.shape[-1]
    net = MODELS[mk](d, dt=dt).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    P = len(Xtr); best = (-9, None); patience = 0
    for ep in range(cfg["epochs"]):
        net.train(); perm = torch.randperm(P)
        for s in range(0, P, 64):
            b = perm[s:s + 64]
            U, a, y, e = Xtr[b].to(device), Atr[b].to(device), Ytr[b].to(device), Ttr[b].to(device)
            sc = net.score(U, a, e) if mk == "text_head" else net.score(U, a)
            loss = rank_loss(sc, y, cfg["loss"], cfg["kappa"]) + cfg["mu"] * net.omega(U.reshape(-1, d))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        r = np.nanmean(within_rho(predict(net, mk, *val[:3], device), val[3], val[4]))
        if r > best[0]:
            best = (r, {k: v.detach().clone() for k, v in net.state_dict().items()}); patience = 0
        else:
            patience += 1
            if patience >= cfg["patience"]: break
    net.load_state_dict(best[1])
    return net, best[0]


@torch.no_grad()
def predict(net, mk, X, A, T, device, bs=256):
    net.eval(); out = []
    for s in range(0, len(X), bs):
        U, a, e = X[s:s + bs].to(device), A[s:s + bs].to(device), T[s:s + bs].to(device)
        out.append((net.score(U, a, e) if mk == "text_head" else net.score(U, a)).float().cpu().numpy())
    return np.concatenate(out, 0)


def ridge_probe(Xtr, Ytr, Xte, lam):
    A = Xtr.reshape(-1, Xtr.shape[-1]).numpy(); b = Ytr.reshape(-1)
    A = np.c_[A, np.ones(len(A))]
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ b)
    Z = Xte.numpy(); return (Z @ w[:-1]) + w[-1]


# ---------------------------------------------------------------------------------------- main ----
def run_set(name, D, args, device, log):
    P = len(D["Y"]); rng = np.random.default_rng(0); perm = rng.permutation(P)
    ntr, nva = int(.60 * P), int(.15 * P)
    tr, va, te = perm[:ntr], perm[ntr:ntr + nva], perm[ntr + nva:]
    Y = D["Y"]

    if name == "compbench":                       # anchor is a sibling -> exclude it, matched for all arms
        mask = np.ones_like(Y, dtype=bool)
        for k in range(P): mask[k, D["anchor"][k]] = False
        A = {r: torch.stack([D["X"][r][k, D["anchor"][k]] for k in range(P)]) for r in REPS}
    else:                                          # anchor is the external photograph
        mask = np.ones_like(Y, dtype=bool); A = D["A"]

    log(f"\n=== {name}: {P} prompts x {Y.shape[1]} candidates   "
        f"train {len(tr)} / val {len(va)} / test {len(te)}  (prompt-disjoint)")

    res = {}
    for r in REPS:                                 # incumbent, no learning
        S = F.cosine_similarity(D["X"][r], A[r].unsqueeze(1), dim=-1).numpy()
        res[f"cos_fixed[{r}]"] = (np.nanmean(within_rho(S[va], Y[va], mask[va])), S)
    if "V" in D:
        res["VQAScore"] = (np.nanmean(within_rho(D["V"][va], Y[va], mask[va])), D["V"])

    for lam in (1, 10, 1e2, 1e3, 1e4):             # ridge probe
        for r in REPS:
            S = np.concatenate([ridge_probe(D["X"][r][tr], Y[tr], D["X"][r][idx], lam)
                                for idx in (np.arange(P),)], 0)
            res[f"probe[{r},lam={lam:g}]"] = (np.nanmean(within_rho(S[va], Y[va], mask[va])), S)

    Yt = torch.tensor(Y, dtype=torch.float32)
    for mk in ("proj_A1", "proj_A2", "text_head"):
        mus = [0.0] if mk == "text_head" else [0.0, 0.01, 0.1, 1.0, 10.0]
        for r in REPS:
            for mu in mus:
                cfg = dict(lr=args.lr, mu=mu, kappa=args.kappa, loss="pair",
                           epochs=args.epochs, patience=args.patience)
                val = (D["X"][r][va], A[r][va], D["T"][va], Y[va], mask[va])
                net, rv = fit(mk, D["X"][r][tr], A[r][tr], Yt[tr], D["T"][tr], val, cfg, device)
                S = predict(net, mk, D["X"][r], A[r], D["T"], device)
                res[f"{mk}[{r},mu={mu:g}]"] = (rv, S)
                log(f"  fit {mk:9s} {r:6s} mu={mu:<5g} val rho={rv:+.4f}")

    # ---- select each family on VAL, report on TEST ------------------------------------------------
    fams = ["cos_fixed", "probe", "proj_A1", "proj_A2", "text_head"] + (["VQAScore"] if "V" in D else [])
    picked = {}
    for f in fams:
        cand = [(v[0], k, v[1]) for k, v in res.items() if k.split("[")[0] == f]
        picked[f] = max(cand, key=lambda t: t[0])

    log(f"\n  {'arm':10s} {'selected on val':28s} {'val rho':>8s} {'TEST rho':>9s} {'TEST headroom':>14s}")
    rows = {}
    for f in fams:
        rv, key, S = picked[f]
        rt = within_rho(S[te], Y[te], mask[te])
        hr = headroom(np.where(mask[te], S[te], -1e9), Y[te])
        rows[f] = rt
        log(f"  {f:10s} {key.split('[')[-1].rstrip(']')[:28]:28s} {rv:+8.4f} {np.nanmean(rt):+9.4f} {hr:13.1f}%")

    # ---- gates ------------------------------------------------------------------------------------
    log(f"\n  paired Wilcoxon on per-prompt TEST rho (Holm across the 3 fitted prompt-blind arms):")
    ref_fixed = rows["cos_fixed"]
    blind = ["probe", "proj_A1", "proj_A2"]
    raw = {}
    for f in blind:
        m = ~np.isnan(rows[f]) & ~np.isnan(ref_fixed)
        raw[f] = wilcoxon(rows[f][m], ref_fixed[m])[1]
    for i, (f, p) in enumerate(sorted(raw.items(), key=lambda t: t[1])):
        delta = np.nanmean(rows[f]) - np.nanmean(ref_fixed)
        a = 0.05 / (len(blind) - i)
        # The gate is one-sided: the projector must BEAT the fixed cosine. A small p with a negative
        # delta is a significant *loss* and must never be labelled a pass.
        verdict = "PASS" if (p < a and delta > 0) else ("WORSE (significant)" if (p < a and delta < 0) else "fail")
        log(f"    {f:10s} vs cos_fixed  delta={delta:+.4f}  p={p:.3e}  Holm alpha={a:.4f}  {verdict}")
    if "VQAScore" in rows:
        for f in blind + ["text_head"]:
            m = ~np.isnan(rows[f]) & ~np.isnan(rows["VQAScore"])
            log(f"    {f:10s} vs VQAScore   delta={np.nanmean(rows[f])-np.nanmean(rows['VQAScore']):+.4f}"
                f"  p={wilcoxon(rows[f][m], rows['VQAScore'][m])[1]:.3e}")
    return {f: float(np.nanmean(v)) for f, v in rows.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="phaseW/feats")
    ap.add_argument("--out", default="phaseW/projector_gate.json")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--kappa", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=15)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.feats)
    lines = []
    def log(s): print(s, flush=True); lines.append(s)

    out = {}
    if (root / "compbench_img.pt").exists():
        out["compbench"] = run_set("compbench", load_compbench(root), args, device, log)
    if (root / "coco_img.pt").exists():
        out["coco"] = run_set("coco", load_coco(root), args, device, log)
    json.dump(out, open(args.out, "w"), indent=1)
    log(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
