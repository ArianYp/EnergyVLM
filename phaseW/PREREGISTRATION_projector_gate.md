# Preregistration — learned DINO projector gate

**Written before a single projector weight is fitted.** Only §7 may be completed afterwards.

## 1. What is being tested

The proposed method ("Ranking Self-Distillation with a Learned DINO Reward") rests on one assumption
that everything else inherits:

> a learned head on frozen DINO features can produce **prompt-conditional** alignment signal.

The proposed reward is `R_phi(x_hat) = cos(g_phi(f(x_hat)), g_phi(f_bar))` with `f` a frozen DINO,
`f_bar = f(x_0)` the cached real-image anchor, and `g_phi` a learned projector. Note that `g_phi`
takes **no text input**: within a round `f_bar` is fixed, so the only thing that varies across the
`m` items is the generation, and the training label is the corruption rank of the prompt that made
it.

If that head cannot out-rank the fixed cosine it is meant to replace, the correction cascade, the
Plackett-Luce distillation, the EMA refresh and the bounded-correction machinery are all moot. This
gate tests it alone, on **already-cached candidates and labels**, with no generation, no student and
no EMA.

## 2. Why this gate is GENEROUS to the proposal

Two ways in which the test gives the projector more than the method would have:

* **Better supervision.** The method trains `g_phi` on `pi_text` — a corruption ladder over
  *prompts*, i.e. a between-prompt ordering. Here it is trained on the **within-prompt gold
  ordering** of 8 candidates for the *same* prompt, which is strictly more informative for the task
  the reward has to perform. 77.6% of CompBench label variance is between-prompt; that is the part
  DINO already reads well (image-only ridge probe: pooled r = +0.394, *above* VQAScore's +0.301).
  The within-prompt 22.4% is the part a selector needs, and it is where the image-only probe
  collapses to rho = +0.062. Training on the harder, more relevant signal can only help.
* **Better anchor.** On CompBench there is no real photograph, so the anchor is the
  **highest-labelled sibling candidate** — an oracle unavailable at deployment. On COCO the anchor
  is the genuine reference photograph, exactly as the method specifies.

Consequence, stated in advance: a failure here is decisive against the method as written; a pass
here is necessary but not sufficient, because the deployed supervision is weaker than the tested
supervision.

## 3. Data (all cached, nothing generated for this gate)

| set | prompts | cands | label | anchor |
|---|---|---|---|---|
| CompBench | 2,398 | 8 | official T2I-CompBench per-prompt score | oracle: best-labelled sibling |
| COCO | 3,000 | 4 | VQAScore (`endpoint_vqa`) | real reference photograph |

Splits are **prompt-disjoint** (all candidates of a prompt fall on the same side).
Train / val / test = 60 / 15 / 25 of prompts, fixed seed 0. All hyperparameters — learning rate,
`mu`, `kappa`, projector rank, epochs — are chosen on **val by within-prompt Spearman**, and every
number in §7 is reported on **test only**.

## 4. Arms

Representations: DINOv2-L and DINOv2-B, CLS and mean-pooled patches (CLS dropped). The
patch-pooled variant is included because it beat CLS on COCO (43.6% vs 39.9% of oracle VQAScore
headroom, job 126439).

| arm | description | text input |
|---|---|---|
| `cos_fixed` | the incumbent, `cos(f(x_hat), f_bar)`, no learning | no |
| `probe` | ridge on `f(x_hat)` -> label, no anchor | no |
| `proj_A1` | `cos(g u, g v)`, PL loss, identity anchor `E‖g(u)-u‖^2` | no |
| `proj_A2` | `cos(g u, g v)`, PL loss, reconstruction anchor `E‖h(g(u))-u‖^2` | no |
| `text_head` | `r_psi(f(x_hat), e(c))`, two-tower + interaction | **yes** |
| `VQAScore` | reference point, not fitted | (is text-conditioned) |

`proj_A1` / `proj_A2` are the proposal's §A1 / §A2. `text_head` is **not** in the proposal; it is
included because binding is a joint property of image and text and a prompt-blind head can only
recover the image-intrinsic component. Its role is diagnostic, declared here so that its inclusion
cannot be read as post-hoc.

## 5. Benchmarks to beat (measured before this gate was written)

Within-prompt Spearman against the official CompBench label, held-out prompts:

| | rho |
|---|---|
| image-only ridge probe (DINOv2-L CLS) | +0.062 |
| `dino_cos`, VQAScore-chosen anchor | +0.086 |
| `dino_cos`, **oracle** anchor | +0.158 |
| VQAScore | **+0.186** |

On COCO, % of oracle VQAScore headroom recovered by fixed cosines: CLS 39.9%, patch-pooled 43.6%.

## 6. Gates — declared with thresholds, before fitting

* **P1 (necessary).** On held-out CompBench prompts the best prompt-blind projector must exceed the
  **oracle-anchored fixed cosine, rho = +0.158**, with a paired Wilcoxon over per-prompt
  coefficients, Holm-corrected across the four fitted prompt-blind arms, alpha = 0.05.
* **P2 (the one that decides the method).** It must additionally reach **VQAScore's rho = +0.186**.
* **C1 (the proposal's own setting).** On COCO with the real-photograph anchor, the best projector
  must exceed the patch-pooled fixed cosine at **43.6%** of oracle VQAScore headroom.

Pre-declared interpretation:

| outcome | conclusion |
|---|---|
| P1 and P2 and C1 pass | the reward is real; build the cascade, with the §7 loss repairs |
| P1 and C1 pass, P2 fails | learning helps, but the reward stays dominated by the scorer it replaces — the cascade is not worth its cost |
| P1 fails | the projector cannot beat a fixed cosine even with oracle supervision and an oracle anchor; the method as written is dead |
| `text_head` passes P2 where every prompt-blind arm fails P1 | the defect is specifically prompt-blindness; repair is to give `g_phi` text input, not to tune `Omega` or the bounded correction |

## 7. Known defects in the distillation half, recorded now so a pass does not paper over them

These are **not** tested by this gate and must be fixed independently of its outcome:

1. Eq. (5) ranks `s_theta^(j)` across items drawn from **different conditionals** `c^(j)`. PL and DPO
   compare responses to the *same* prompt; the EMA log-ratio only partially normalises this.
2. Phase-FP measured that this loss family **degrades the loser rather than improving the winner**.
   Here the loser is a *correct* rendering of a corrupted-but-legitimate prompt, so the gradient
   teaches the model to render such prompts badly.
3. `pi_text` asserts a total order over corruption severity without saying how it is built. The only
   construction we have validated is Phase-FP's **dose ladder** (severity = number of atomic edits).
   Absent that, use the partial order (positive > each negative, no order among negatives), which
   collapses PL to `m-1` pairwise terms.
4. The Phase-K C2 edit engine emits **null edits 8% of the time (25% on texture)**, so `pi_text` has
   label noise at source, independent of the bad-positive problem that §B addresses.
5. `beta` will be inert: gradient clipping fired on 100% of steps in our preference runs, so the
   temperature does not scale the update. Tune the clip threshold, not `beta`.

## 8. RESULT

Completed 2026-09-01, job 128103 (rerun 128125 after a reporting fix, see deviations). Held-out
TEST prompts, all selection done on VAL.

### CompBench, oracle anchor (601 test prompts)

| arm | selected on val | val rho | **TEST rho** | TEST headroom |
|---|---|---|---|---|
| `cos_fixed` | b_pat | +0.1815 | **+0.1720** | 8.7% |
| `proj_A1` | b_pat, mu=0 | +0.1913 | +0.1723 | 14.6% |
| `proj_A2` | b_pat, mu=0.1 | +0.1884 | +0.1602 | 13.2% |
| `probe` | b_pat, lam=1 | +0.0806 | +0.0583 | 2.0% |
| `text_head` | l_pat, mu=0 | +0.1353 | +0.0944 | 4.6% |
| VQAScore | — | +0.2241 | +0.1621 | 15.5% |

Paired Wilcoxon vs `cos_fixed`: `proj_A1` delta **+0.0003**, p=0.685; `proj_A2` delta -0.0118,
p=0.256; `probe` delta -0.1137, p=3.6e-06 (a significant **loss**).

### COCO, real-photograph anchor (750 test prompts)

| arm | selected on val | val rho | TEST rho | **TEST headroom** |
|---|---|---|---|---|
| `cos_fixed` | b_pat | +0.2370 | +0.2430 | **44.5%** |
| `proj_A2` | b_pat, mu=10 | +0.2431 | +0.2601 | 44.9% |
| `proj_A1` | l_pat, mu=1 | +0.2518 | +0.2133 | 43.6% |
| `probe` | b_pat, lam=100 | +0.2398 | +0.2328 | 40.8% |
| `text_head` | l_pat, mu=0 | +0.2265 | +0.2050 | 34.3% |

Paired Wilcoxon vs `cos_fixed`: `proj_A2` delta +0.0172, p=0.230; `proj_A1` delta -0.0297, p=0.116;
`probe` delta -0.0102, p=0.561. Nothing clears Holm.

### Verdict

* **P1 FAILS.** No prompt-blind projector beats the fixed cosine. `proj_A1`'s advantage is
  **+0.0003** — three ten-thousandths of a rank correlation, p=0.685.
* **P2** is moot. (`proj_A1` does numerically tie VQAScore, delta +0.0102, p=0.793, but only via an
  oracle anchor it would never have.)
* **C1 FAILS.** The best COCO projector gains +0.4 points of headroom, p=0.230.

Per the §6 table, row 3: **the projector cannot beat a fixed cosine even given oracle supervision
and an oracle anchor, so the method as written is dead.** The positive control rules out a code
defect: on planted anchor-mediated data the same fitting code takes the fixed cosine from rho=+0.431
to +0.97.

**The predicted repair also fails, and this was my prediction, not the proposal's.** `text_head` was
added to §4 on the argument that prompt-blindness was the specific defect. It is the *worst* fitted
arm on COCO (34.3% vs 44.5%) and significantly below VQAScore on CompBench (-0.0677, p=0.0079), and
it loses on val as well as test, so this is not test-split noise — it did not learn. Bounded claim:
one architecture (two-tower + multiplicative interaction), SD3.5 pooled + sequence-mean text
embeddings, 1,438 training prompts. That is thin supervision for a text-conditioned head, so this
refutes *my* proposed repair as specified, not text conditioning in general.

**Byproduct that stands on its own: the CLS-vs-patch question is settled for CompBench too.** The
fixed cosine on full mean-pooled DINOv2-B patches reaches rho=+0.1720, above the +0.158 that
DINOv2-L CLS reached in §5. The earlier apparent CLS win was the rank-64 PCA truncation, exactly as
suspected. Patch-pooled is now better on both datasets, measured with matched features.

Also worth recording: on this test split the oracle-anchored patch cosine (+0.1720) slightly exceeds
VQAScore (+0.1621) on rank correlation while losing badly on headroom (8.7% vs 15.5%). Rank
correlation over 7 candidates and argmax quality are different questions and they disagree here;
headroom is the one that matters for selection.

### Deviations from the plan above

1. §6 says "four fitted prompt-blind arms"; there are **three** (`probe`, `proj_A1`, `proj_A2`).
   Holm was applied across three. This makes the correction slightly *less* conservative than
   written, and no arm came close to significance, so it changes nothing.
2. Job 128103's log printed `probe vs cos_fixed ... PASS` for a delta of **-0.1137**: the verdict
   line tested the p-value without the direction. The gate is one-sided. Fixed in
   `phaseW/projector_gate.py` and rerun as 128125; no reported conclusion depended on it, but the
   raw log of 128103 contains that mislabel.
