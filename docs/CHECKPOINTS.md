# Using the distilled students

Every checkpoint below is a **4-step, guidance-free** student of `stabilityai/stable-diffusion-3.5-medium`.
It is the SD3.5-Medium transformer only (2B parameters, 909 tensors, fp32, 9.9 GB); the VAE and the three
text encoders are unchanged, so you load the stock pipeline and swap the transformer weights in.

Trained and evaluated at **512x512**. They are not fine-tuned for 1024 and score worse there.

## Which checkpoint

Paths are on Lustre and group-readable (`team361`).

```
/lustre/scratch126/cellgen/lotfollahi/ha11/EnergyVLM/checkpoints/phaseW/
```

| use | run directory (append `/checkpoint_avg_last5.pt`) |
|---|---|
| **default: best student** | `phaseW_CD_dinop_hard_3k-k10-hp1-acc4_s0_153472` |
| two more seeds of the same recipe, for variety when cherry-picking | `..._s1_153828`, `..._s2_153836` |
| baseline: same recipe, no scored selection | `phaseW_B2_3k-k10-hp1-acc4_s0_153468` |
| the arm reported in the paper (K=8 cache) | `phaseW_CD_dinop_hard_3k-rewRi-s16-hp1-acc4_s0_145176` |

Take `checkpoint_avg_last5.pt` (an average of the last checkpoints), not `checkpoint_final.pt`: it is
~0.002 CompBench better and visibly more stable.

T2I-CompBench (official protocol, 10 images per prompt, three seeds) for the default checkpoint:
0.4965 on the scheduler grid, **0.5034 on grid A** (below). The 28-step guided teacher scores 0.5053.

There is also `phaseW_CD_bench_hard_bench-k10-hp1-acc4_s0_155631`, trained on T2I-CompBench's own
training prompts. It scores higher on that benchmark but is tuned to those prompt templates — do not
use it for figures or for editing work.

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
