# Using the distilled students

Every checkpoint below is a **4-step, guidance-free** student of `stabilityai/stable-diffusion-3.5-medium`.
It is the SD3.5-Medium transformer only (2B parameters, 909 tensors, fp32, 9.9 GB); the VAE and the three
text encoders are unchanged, so you load the stock pipeline and swap the transformer weights in.

Trained and evaluated at **512x512**. They are not fine-tuned for 1024 and score worse there.

## Which checkpoint

Paths are on Lustre and group-readable (`team361`). Every row below is a run directory under

```
/lustre/scratch126/cellgen/lotfollahi/ha11/EnergyVLM/checkpoints/phaseW/
```

and the file to load is **`<run>/checkpoint_avg_last5.pt`** (an average of the run's last checkpoints;
~0.002 CompBench better than `checkpoint_final.pt` and visibly more stable). Seeds of one recipe differ
only in their training seed; use them for variety when cherry-picking.

Arms: **selection** = the student distils the DINOv2-scored teacher trajectory; **projector reward** =
the paper's reward through the refreshed latent-to-DINO projector (lambda 80, refreshed every 100
updates for 16 steps); **naive** = a random trajectory, no reward. Scores: T2I-CompBench on the 2,398
held-out val prompts at 4 steps, guidance 1, scheduler grid, one image per prompt (`10:` = the official
ten-images-per-prompt protocol), and GenEval2. The 28-step guided teacher scores 0.5053.

### 3k COCO captions, converged schedule — the default

| use | run directory | CompBench | GenEval2 |
|---|---|---|---|
| **default: selection + projector reward, K=10 teacher grid** | `phaseW_CD_dinop_hard_3k-k10-hp1-acc4_s0_153472` | 0.4951 (10: 0.4965; **grid A** 10: **0.5034**) | 0.226 |
| the same, seeds 1 and 2 | `phaseW_CD_dinop_hard_3k-k10-hp1-acc4_s1_153828`, `..._s2_153836` | 0.4970 / 0.4938 (10: 0.4965 / 0.4953) | 0.232 / 0.236 |
| naive, K=10 (the default's baseline) | `phaseW_B2_3k-k10-hp1-acc4_s0_153468`, `..._s1_153824`, `..._s2_153832` | 0.4877 / 0.4910 / 0.4922 (10: 0.4920 / 0.4902 / 0.4927) | 0.232 / 0.230 / 0.230 |
| the paper's arm: selection + projector reward, K=8 grid | `phaseW_CD_dinop_hard_3k-rewRi-s16-hp1-acc4_s0_145176`, `..._acc8_s1_151403`, `..._acc8_s2_151361` | 0.4877 / 0.4874 / 0.4884 (10: 0.4898 / 0.4894 / 0.4905) | 0.230 / 0.228 / 0.236 |
| naive, K=8 (the paper's baseline) | `phaseW_B2_3k-hp1-acc4_s0_145172`, `..._acc8_s1_151349`, `..._acc8_s2_151357` | 0.4860 / 0.4857 / 0.4852 (10: 0.4852 / 0.4861 / 0.4872) | 0.230 / 0.228 / 0.224 |

### 118k COCO captions

None of these beats the 3k default on CompBench (16 passes over 3k captions beat 2 passes over 118k,
`docs/bench/`, README); the first row has the best fidelity of any student.

| use | run directory | CompBench | GenEval2 |
|---|---|---|---|
| **best 118k student: selection + exact DINO reward** — the reward through VAE decode + DINOv2 itself (lambda 15.5), not the projector; constant LR, 56,974 updates | `phaseW_CD_dinop_hard_118k-rewX_s0_138926`, `..._s1_138931`, `..._s2_138936` | 0.4933 / 0.4962 / 0.4896 | 0.239 / 0.217 / 0.234 |
| the paper's arm at 118k: selection + projector reward, constant LR, 56,974 updates | `phaseW_CD_dinop_hard_118k-rewRi-s16_s0_145096`, `..._s1_145100`, `..._s2_145104` | 0.4832 / 0.4753 / 0.4845 | 0.226 / 0.204 / 0.220 |
| selection + projector reward, converged schedule (cosine, batch 16, 14,244 updates), K=10 grid | `phaseW_CD_dinop_hard_118k-k10-hp1-acc4_s0_154567` (naive control `phaseW_B2_118k-k10-hp1-acc4_s0_154563`) | 0.4815 (naive 0.4776) | 0.227 (0.230) |
| selection only, no reward, constant LR | `phaseW_CD_dinop_hard_118k_s0_128710`, `..._s1_130350`, `..._s2_130354` | 0.4865 / 0.4821 / 0.4843 (10: 0.4878 / 0.4838 / 0.4832) | 0.221 / 0.212 / 0.215 |
| VQAScore-selected trajectories, no reward, constant LR (best 118k GenEval2) | `phaseW_B4_118k_s0_128711` | 0.4827 (10: 0.4835) | 0.237 |
| naive, constant LR (the 118k baseline) | `phaseW_B2_118k_s0_128712`, `..._s1_130342`, `..._s2_130346` | 0.4642 / 0.4709 / 0.4653 (10: 0.4704 / 0.4723 / 0.4729) | 0.206 / 0.205 / 0.197 |

Fidelity at 118k (5,000 COCO captions, `phaseW/fidelity_118k_rewX_report.md`): exact-reward CMMD
0.69 / 0.68 / 0.68, FID 30.5-31.3, precision 0.56-0.57; selection-only CMMD 0.78-0.79; naive 0.83-0.84.

### T2I-CompBench train prompts (benchmark-trained; `docs/bench/`)

Distilled on the benchmark's own 5,559 training prompts, the trajectory chosen by the official
evaluator of each prompt's category. **Highest benchmark scores in the project, but tuned to those
prompt templates ("a red bench and a green car"): use them for benchmark comparisons, not for figures
or editing work.**

| use | run directory | CompBench | GenEval2 |
|---|---|---|---|
| **best benchmark score: evaluator-argmax selection, no reward** | `phaseW_CD_bench_hard_bench-k10-hp1-acc4_s0_155631`, `..._s1_155880`, `..._s2_155888` | 0.5012 / 0.4986 / 0.5011; seed 0 on grid A 0.5052, **seed 0 at 8 steps 0.5107** (above the teacher) | 0.231 / 0.236 / 0.237 |
| evaluator-argmax + projector reward (a null here: no photograph to point at) | `phaseW_CD_bench_hard_bench-k10-rew-hp1-acc4_s0_158197` | 0.5011 | 0.223 |
| random pick (the in-domain control) | `phaseW_B2_bench-k10-hp1-acc4_s0_155627`, `..._s1_155876`, `..._s2_155884` | 0.4892 / 0.4880 / 0.4893 | 0.220 / 0.222 / 0.227 |
| random pick + projector reward | `phaseW_B2_bench-k10-rew-hp1-acc4_s0_158201` | 0.4883 | 0.224 |

Every number above is read from the run's own evaluation record (`alignment.json` whose `ckpt` is that
file) and every reward setting from the run's `args.json`.

## Sampling

Three things differ from stock SD3.5 and all three matter:

1. **4 steps.**
2. **Guidance 1.0** (no CFG). The student has guidance internalised; sampling it at cfg 7 produces garbage.
3. **Grid A** (optional, free, recommended): sample on sigmas `1, 0.882788, 0.693793, 0.337972, 0`
   instead of the scheduler's 4-step grid. Worth +0.006 CompBench and better FID/CMMD, no retraining.
   The last step is a jump to the clean image from sigma 0.338.

   Watch out: `pipe(..., sigmas=...)` applies the scheduler's shift (3.0) to whatever you pass, so
   hand it the **pre-shift** values `1, 0.715142, 0.430282, 0.145423` and the scheduler ends up on
   grid A. `eval/generate.py --sigmas` takes the post-shift values directly instead. Print
   `pipe.scheduler.sigmas` once after a call if you want to be sure which grid you are on.

```python
import torch
from diffusers import StableDiffusion3Pipeline

CKPT = ("/lustre/scratch126/cellgen/lotfollahi/ha11/EnergyVLM/checkpoints/phaseW/"
        "phaseW_CD_dinop_hard_3k-k10-hp1-acc4_s0_153472/checkpoint_avg_last5.pt")

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16).to("cuda")
# torch_dtype alone leaves CLIP's text_projection in fp16 and prompt encoding then dies with
# "expected mat1 and mat2 to have the same dtype: Half != BFloat16". Cast every component.
for m in (pipe.transformer, pipe.vae, pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3):
    m.to(dtype=torch.bfloat16).eval()

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
missing, unexpected = pipe.transformer.load_state_dict(ck["model"], strict=False)
assert not missing and not unexpected            # fail loudly if the checkpoint is not this model
pipe.transformer.to(dtype=torch.bfloat16).eval()
del ck

img = pipe("a white piano and a black bench", num_inference_steps=4,
           guidance_scale=1.0, height=512, width=512,
           sigmas=[1.0, 0.715142, 0.430282, 0.145423],   # -> grid A after the shift; omit for the scheduler grid
           generator=torch.Generator("cuda").manual_seed(0)).images[0]
```

`ck` also carries `step` and `variant`; nothing else is needed at inference.

## More denoising steps

The student is distilled to jump from four specific noise levels, so running it longer is not the
free win it is for an ordinary diffusion model. Measured on the held-out pool, one image per prompt,
guidance 1.0, the scheduler's own grid at each step count:

| steps | CompBench | GenEval2 | colour | texture | 2D-spatial | 3D-spatial | numeracy |
|---|---|---|---|---|---|---|---|
| 2 | 0.1693 | 0.033 | 0.324 | 0.248 | 0.005 | 0.036 | 0.085 |
| **4** (trained) | 0.4951 | **0.2257** | **0.809** | **0.741** | 0.243 | 0.333 | 0.572 |
| **8** | **0.5001** | 0.2060 | 0.799 | 0.733 | **0.258** | **0.370** | 0.582 |
| 16 | 0.4955 | 0.1966 | 0.792 | 0.717 | 0.243 | 0.362 | **0.586** |
| 28 | 0.4873 | 0.1911 | 0.784 | 0.710 | 0.242 | 0.339 | 0.576 |

Below 4 the model collapses: 2 steps is blurred colour fields, not images.

Above 4 there is a **real trade, not a decline**. Eight steps gives the best CompBench of any setting,
and the gain sits exactly where the 4-step student is weakest: 3D-spatial 0.333 -> 0.370, 2D-spatial
0.243 -> 0.258, numeracy 0.572 -> 0.582. It costs attribute binding (colour and texture each lose
~0.01) and GenEval2, which drops from 0.226 to 0.206 because it zeroes a prompt when any single atom
is wrong. `docs/figs/steps_sweep.jpg` shows the same effect by eye: more steps add structure and
detail, and push colour and contrast around.

Past 8 there is nothing left to win. 16 keeps losing attributes for no gain, 28 is worse everywhere.

Practical advice:

- **Anything you measure or report: 4 steps.** It is the trained operating point, the best GenEval2
  and the best colour and texture, and it is what every number in this repository describes.
- **Figures, especially scenes with spatial structure or counted objects: try 8.** Then check by eye
  that the colours and materials in your prompt are still right, since that is what you are paying with.
- **Never above 8**, and never below 4.
- Want quality without the trade? Sample the **teacher** (`--checkpoint base --cfg 7.0 --steps_list 28`,
  CompBench 0.5053) and accept 56 forward passes instead of 4 or 8.

All rows: best checkpoint, seed 0, held-out pool, one image per prompt, guidance 1.0, the scheduler's
own grid at each step count (grid A is defined for 4 steps only).

## Batch generation (many prompts / many seeds)

`eval/generate.py` in this repository does prompt sharding across GPUs, deterministic seeding and
decoding. Prompts are a json list of `{"idx": int, "category": str, "prompt": str}`.

```bash
torchrun --standalone --nproc-per-node=4 eval/generate.py \
  --out_root out/figs --label ours --checkpoint "$CKPT" \
  --cfg 1.0 --steps_list 4 --prompts_json my_prompts.json \
  --n_seeds 16 \                                   # 16 images per prompt, for cherry-picking
  --sigmas 1,0.882788,0.693793,0.337972,0
```

Images land at `out/figs/images/ours/p{idx:05d}/s4/cand{j}.png`. Candidate `j` of prompt `idx` always
uses the same noise, so a picture you like is reproducible from its `(idx, j)`.

## Cherry-picking for figures

Generate 16-32 seeds per prompt as above, then rank them instead of eyeballing everything. The scorer
we used is VQAScore (`clip-flant5-xxl`), which agrees with human judgement on prompt following far
better than CLIP does:

```python
import sys
sys.path.insert(0, "third_party/t2v_metrics")
from common import t2v_compat     # REQUIRED before importing t2v_metrics, see below
import t2v_metrics
score = t2v_metrics.VQAScore(model="clip-flant5-xxl")
s = score(images=["cand0.png", "cand1.png"], texts=["a white piano and a black bench"])
```

`common/t2v_compat.py` stubs two optional backends (a Gemini API scorer and a video decoder) that
`t2v_metrics` imports unconditionally and that crash on our environment. Import it first or the
import fails.

For a picture that is simply *pretty* rather than prompt-faithful, rank by aesthetic score instead;
we did not use one, so pick your own.

## FlowEdit and other editing work

SD3.5-Medium is a rectified-flow model and the student keeps that parameterisation, so FlowEdit
applies unchanged. Two adjustments:

- Set the **student's** sampling to 4 steps at guidance 1.0. Where FlowEdit expects a source and a
  target guidance scale, the student's internalised guidance replaces the target one; start with
  `src_guidance=1.0, tgt_guidance=1.0` and raise only the source if the edit is too weak.
- FlowEdit's step count is its own; with a 4-step student, use its `n_avg`/`n_max` on the same 4-step
  grid rather than the 28-step default, otherwise it will drive the student at sigmas it never saw.

If the edit quality disappoints, compare against the **teacher** (`--checkpoint base --cfg 7.0
--steps_list 28`) before blaming the student: a 4-step model has less room for an inversion-free edit
to work in.

## Gotchas

- **cfg 1.0, always** for students; `7.0` only for the base/teacher (`--checkpoint base`).
- The checkpoints are fp32 on disk; cast to bf16 after loading, as above.
- `strict=False` with the two assertions is deliberate: it catches a checkpoint from a different model
  instead of silently loading a partial state dict.
- 512x512. At 1024 the same checkpoint drops ~0.02 CompBench.
- The environment we used is `/software/cellgen/team361/ha11/envs/nichejepa`.
