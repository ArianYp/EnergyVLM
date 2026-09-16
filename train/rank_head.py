#!/usr/bin/env python3
"""Prompt-aware SELECTION, stage 2 (docs/rank/): train the ranking head on true-DINO features of teacher images.

Partial-order loss per caption (positive above each negative), both sides through the head, on the
embeddings of data/build_rank_pairs.py. Validation on the held-out captions: pairwise accuracy raw vs shaped, the
shuffled-anchor control, per family. Saves head_dino.pt.

    python3 train/rank_head.py --dir cache/rank_pairs
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train.rank_utils import RankHead, rank_rows, rows_loss  # noqa: E402


def metrics(head, recs, shuffle_anchor=False):
    acc, accr, fam, famr, marg = [], [], defaultdict(list), defaultdict(list), []
    with torch.no_grad():
        for i, r in enumerate(recs):
            u = (recs[(i + 1) % len(recs)]["u"] if shuffle_anchor else r["u"])[None]
            R = rank_rows(head, r["e_pos"][None], r["e_negs"][:, None, :], u, False)[0]
            Rr = rank_rows(None, r["e_pos"][None], r["e_negs"][:, None, :], u, False)[0]
            for j, n in enumerate(r["negatives"], start=1):
                acc.append(float(R[0] > R[j])); accr.append(float(Rr[0] > Rr[j]))
                fam[n["family"]].append(float(R[0] > R[j])); famr[n["family"]].append(float(Rr[0] > Rr[j]))
            marg.append(float(R[0] - R[1:].mean()))
    mean = lambda v: sum(v) / max(len(v), 1)  # noqa: E731
    return {"acc_shaped": mean(acc), "acc_raw": mean(accr), "margin_shaped": mean(marg), "n_pairs": len(acc),
            "per_family": {f: {"shaped": mean(v), "raw": mean(famr[f]), "n": len(v)} for f, v in fam.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="cache/rank_pairs")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--kappa", type=float, default=0.1)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); random.seed(args.seed)
    d = Path(args.dir)
    train = torch.load(d / "pairs_train.pt", weights_only=False); held = torch.load(d / "pairs_heldout.pt", weights_only=False)
    # a caption-level validation split of the training pool too (the head's own captions are the ones it re-scores)
    random.shuffle(train); n_val = max(50, len(train) // 10); val, fit = train[:n_val], train[n_val:]
    head = RankHead(768, args.width)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    best, best_state, log = -1.0, None, []
    m0 = {"heldout": metrics(head, held), "val": metrics(head, val)}
    print(f"[head] init: heldout raw {m0['heldout']['acc_raw']:.3f} shaped {m0['heldout']['acc_shaped']:.3f} | val raw {m0['val']['acc_raw']:.3f}", flush=True)
    for ep in range(args.epochs):
        random.shuffle(fit); tot = 0.0
        for b in range(0, len(fit), args.batch):
            loss = 0.0
            for r in fit[b:b + args.batch]:
                R = rank_rows(head, r["e_pos"][None], r["e_negs"][:, None, :], r["u"][None], True)
                loss = loss + rows_loss(R, args.kappa)
            loss = loss / len(fit[b:b + args.batch])
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step(); tot += float(loss)
        mv, mh = metrics(head, val), metrics(head, held)
        log.append({"epoch": ep + 1, "loss": tot / max(1, len(fit) // args.batch), "val": mv["acc_shaped"], "heldout": mh["acc_shaped"]})
        print(f"[head] epoch {ep + 1:3d} loss {log[-1]['loss']:.4f} val shaped {mv['acc_shaped']:.3f} (raw {mv['acc_raw']:.3f}) heldout shaped {mh['acc_shaped']:.3f} (raw {mh['acc_raw']:.3f})", flush=True)
        if mv["acc_shaped"] > best:
            best, best_state = mv["acc_shaped"], {k: v.clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    final = {"heldout": metrics(head, held), "heldout_shuffled_anchor": metrics(head, held, shuffle_anchor=True),
             "val": metrics(head, val), "train_fit": metrics(head, fit), "init": m0, "log": log, "args": vars(args),
             "n": {"fit": len(fit), "val": len(val), "heldout": len(held)}}
    torch.save({"head": head.state_dict(), "width": args.width, "metrics": final}, d / "head_dino.pt")
    json.dump(final, open(d / "head_dino_metrics.json", "w"), indent=1)
    h, s = final["heldout"], final["heldout_shuffled_anchor"]
    print(f"\n[head] HELD-OUT: raw {h['acc_raw']:.3f} -> shaped {h['acc_shaped']:.3f} (margin {h['margin_shaped']:+.4f}, {h['n_pairs']} pairs); "
          f"shuffled anchor shaped {s['acc_shaped']:.3f}")
    for f, v in sorted(h["per_family"].items()):
        print(f"   {f:8s} raw {v['raw']:.3f} shaped {v['shaped']:.3f} (n {v['n']})")
    print("wrote", d / "head_dino.pt")


if __name__ == "__main__":
    main()
