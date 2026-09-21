# Share-CoT: the official T2I-CompBench++ non-spatial evaluator

`eval/compbench.py` maps `non_spatial -> clipscore`. T2I-CompBench++ TABLE XIII -- and every paper
that copies its baseline rows, including CTCal Table 1 -- reports the **Non-Spatial** column with
**Share-CoT** (ShareGPT4V-7B answering with a chain of thought), not CLIPScore. The two are different
metrics on different scales (Share-CoT ~0.75-0.79, CLIPScore ~0.31) and CLIPScore barely separates
models at all (0.298-0.320 over ++'s 12 systems), which is why ++ replaced it. See
`docs/lit/ctcal_alignment_audit.md` sections 1.1 and 3.1.

`eval/sharecot_nonspatial.py` makes the vendored official evaluator runnable here. It adds a new
`scores.json` directory next to the existing ones; nothing existing is modified.

```
eval/sharecot_nonspatial.py          wrapper: stage -> run the official Share_eval.py -> collect scores.json
eval/compat/seeded_share_eval.py     execs the official script verbatim with the RNGs seeded
eval/compat/sharecot_requirements.txt  pip freeze of the environment below
scripts/sharecot_score.lsf           one eval dir, 1 GPU, ~20 min for 300 images
cache/sharecot_runroot/              cwd for the official script (holds the vision-tower symlink; created on first run)
```

## Environment

The evaluator is LLaVA-1.5-era code and cannot share the training environment (torch 2.7 /
transformers 4.49). It lives in its own conda env, on this cluster
`/software/cellgen/team361/ha11/envs/sharecot` (6.7 GB; `SHARECOT_BIN` in the launcher):

```bash
export PYTHONNOUSERSITE=1
conda create -p <env> python=3.10 -y
P=<env>/bin/python
# cu118, not the PyPI default cu117 build: H100 / H200 need sm_90 kernels
$P -m pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
$P -m pip install numpy==1.26.4 transformers==4.31.0 tokenizers==0.13.3 sentencepiece==0.1.99 \
    accelerate==0.21.0 huggingface_hub==0.16.4 einops==0.6.1 einops-exts==0.0.4 timm==0.6.13 \
    scikit-learn==1.2.2 shortuuid ftfy protobuf matplotlib tqdm spacy==3.7.5 "click<8.2"
cd third_party/T2I-CompBench/MLLM_eval/ShareGPT4V-CoT_eval
$P -m pip install -e . --no-deps          # installs the vendored `llava` package as `share4v`
$P -m spacy download en_core_web_sm        # Share_eval.py's noun-chunk splitter
```

**`PYTHONNOUSERSITE=1` is mandatory** wherever a user-site `~/.local/lib/python3.10/site-packages`
precedes the env's on `sys.path` (it does on this cluster and carries a newer pydantic /
typing_extensions / requests). Without it the env is silently not the env. The launcher exports it;
so does the wrapper for its subprocess.

Deviations from upstream's install recipe (`T2I-CompBench/Readme.md:169-233`):

| upstream | here | why |
|---|---|---|
| `pip install -e .` and `-e ".[train]"` (full pin set) | `pip install -e . --no-deps` + the runtime pins above | the pinned `gradio==4.5.0`, `xformers==0.0.21`, `bitsandbytes==0.41.0`, `peft==0.4.0`, `deepspeed` are unused by `Share_eval.py` (verified: its only third-party imports are torch, transformers, PIL, spacy, matplotlib, tqdm) and `xformers==0.0.21` would drag in a cu117 torch |
| `pip install flash-attn` | skipped | the inference path never touches it |
| `torch==2.0.1` from PyPI | `torch==2.0.1+cu118` | the PyPI default is cu117, which has no sm_90 kernels |
| `git lfs clone` the vision tower into `<script dir>/Lin-Chen/` | symlink in `cache/sharecot_runroot/Lin-Chen/`, weights in the HF cache | keeps the vendored `T2I-CompBench` tree clean -- see "Vision tower" |

## Weights

| repo | size |
|---|---|
| `Lin-Chen/ShareGPT4V-7B` | 13 GB |
| `Lin-Chen/ShareGPT4V-7B_Pretrained_vit-large336-l12` | 580 MB |

Both are downloaded into `HF_HOME` once (`huggingface-cli download <repo>`); the wrapper resolves them
with `local_files_only=True`.

**Vision tower.** The 7B checkpoint contains **no** `vision_tower.*` weights (0 such keys in
`pytorch_model.bin.index.json`), and `llava/model/multimodal_encoder/builder.py` accepts a tower only
if `os.path.exists(name)` or the name starts with `openai` / `laion` -- an HF id like
`Lin-Chen/ShareGPT4V-7B_...` raises `ValueError: Unknown vision tower`. Upstream solves this by
`git clone`-ing the tower into `Lin-Chen/` beside the script. The wrapper instead builds that same
relative layout under `cache/sharecot_runroot/` (`SHARECOT_RUNROOT` to move it) and runs the official
script with that cwd, so the ViT weights and its `preprocessor_config.json` come from the HF cache and
the vendored repo keeps a clean git tree. The symlink is (re)created automatically.

## Running

```bash
bsub -env "all,EVAL_DIR=out/eval/eval_<label>,LABEL=<label>,NIMG=1" < scripts/sharecot_score.lsf
```

| env var | meaning |
|---|---|
| `EVAL_DIR` | eval root containing `compbench/images/<LABEL>/p*/s*/cand*.png` (`scripts/eval_alignment.lsf`) |
| `LABEL` | the directory name under `compbench/images/` |
| `NIMG` | images per prompt: `1` -> `cand0.png`; `10` -> the official protocol |
| `STEPS` | optional, inferred from the `s<N>` level of the image tree |
| `SEED` | optional, default 0 |
| `PROMPTS`, `CATEGORY` | optional, another prompt pool and its category name; the pool's idx numbering MUST match the image tree (`pools/eval` and `pools/bench_train` number differently: the wrong pool silently scores every prompt against an unrelated image) |
| `LIMIT`, `OUT` | optional, for smoke tests |
| `SHARECOT_BIN` | the env's `bin/` (default: the cluster path above) |

Direct (inside the env, on a GPU node):

```bash
python eval/sharecot_nonspatial.py --evaldir out/eval/eval_base --label base --images cand0.png
```

Output: **`<EVAL_DIR>/compbench_scores/<LABEL>_s<STEPS>_non_spatial_sharecot/`**

```
manifest.json                 the staging manifest (same shape as eval/compbench.py's)
samples/<prompt>_<qid>.png    symlinks in the evaluators' filename contract
sharegpt4v/total.json         upstream: CoT description + answer per image
sharegpt4v/score.json         upstream: raw answer per image
sharegpt4v/vqa_result.json    upstream: [{question_id, answer(20..100)}]  <- the official numbers
sharegpt4v/score.txt          upstream: "total:... num:... avg:..."
scores.json                   ours, the schema of every other compbench_scores/*/scores.json, plus "evaluator": "sharegpt4v_cot"
```

`scores.json` carries the usual `dir / evaluator / steps / images_root / n / mean / per_category /
per_prompt / images_per_prompt / per_image` keys (`per_prompt` score = mean over that prompt's
images), plus Share-CoT provenance: `official_script`, `official_category`, `cot`, `model_path`,
`model_name`, `seed`, `temperature`, `top_p`, `score_scale`, `mean_raw`, `n_parse_failed`,
`n_no_answer`, `run_seconds`, `seconds_per_image`. `per_image` rows additionally carry `score_raw`
(the official 20..100 value), `answer` and `description` (the CoT text), so every score is auditable.

## How the score is produced (and the two traps)

1. `Share_eval.py --cot` first asks for a <= 50-word description of the image, then appends the
   description to the context and asks for a 1-5 verdict "in JSON format with the keys score,
   explanation", at `temperature=0.2, top_p=0.7, max_new_tokens=512`.
2. Upstream maps `{1:20, 2:40, 3:60, 4:80, 5:100}` and averages -> `score.txt` avg ~75. Published
   tables are that / 100 (SD1.4 0.7487, SD2 0.7567, SD3 0.7782), so `scores.json` stores
   `score = raw/100` and keeps `score_raw`.

**Trap 1 -- the category is `action`, not `non_spatial`.** The official script has no `non_spatial`
branch; CompBench's non-spatial prompts are action-relation prompts (`examples/dataset/non_spatial_val.txt`)
and both MLLM evaluators call that category `action` (`Share_eval.py:131,204`, `gpt4v_eval.py:162`).
`--category non_spatial` falls through every branch and dies with `NameError: query`. The wrapper
passes `action` by default (`--official_category`).

**Trap 2 -- an unparseable answer scores 1.0.** Upstream's aggregator initialises `score_i = 100` and
`continue`s on a `json.loads` failure, so a malformed answer is indistinguishable from an honest
verdict of 5. This is upstream behaviour and is baked into every published Share-CoT number, so it is
reproduced exactly -- but `collect` re-derives the scores from the raw answers with byte-identical
logic, asserts agreement with upstream's `vqa_result.json`, and reports `n_parse_failed` /
`n_no_answer` so the contamination is visible. Check those fields before quoting a mean (0 of 300 in
every run so far).

## Deviations from the official evaluation script

* **`Share_eval.py` itself is not modified.** `eval/compat/seeded_share_eval.py` seeds `random` /
  `numpy` / `torch` from `SHARECOT_SEED` (default 0) and then `runpy.run_path`s the official file
  verbatim with `run_name="__main__"`. Upstream seeds nothing while sampling at temperature 0.2, so
  its numbers are not reproducible run to run; ours are, per GPU and library build. Seeded-but-sampled
  is still stochastic across seeds -- quote the seed.
* **cwd** is `cache/sharecot_runroot/` instead of the script's own directory (see "Vision tower").
* **Aggregation** is ours: upstream's `vqa_result.json` is joined back onto the manifest by
  `question_id` (which `Share_eval.py` parses out of the filename, so the join is exact -- unlike
  BLIP / CLIPScore, which emit a positional id) and averaged per prompt. Upstream only prints a flat
  mean over images.
* Everything else -- prompt template, CoT turn, temperature, top_p, max_new_tokens, the 1-5 rubric,
  the 20..100 map, the parse -- is upstream's.

## Protocol caveats (unchanged by this script)

* **1 image per prompt** by default, not the official 10 (`Readme.md:310`). Set `NIMG=10` once
  `cand1..9.png` exist (`IMAGES_PER_PROMPT=10` on `scripts/eval_alignment.lsf`); per-prompt variance is
  ~3.2x larger at N=1.
* **512 px** generation vs the 1024 px of ++'s SD3 rows. Share-CoT is far less resolution-sensitive
  than BLIP-VQA / UniDet, but the caveat stands.
* 300 `non_spatial_val` prompts, matching the official split.

## Timings (H200, 1 GPU, fp16, batch 1)

| stage | cost |
|---|---|
| model load | ~30 s |
| per image | **3.4 s** (two generations: CoT description + verdict) |
| 300 prompts x 1 image | 17 min wall |
| 300 prompts x 10 images (official) | ~2.9 h -- inside `-W 4:00`, but not by much |

`-W 4:00`, `-n 4`, 64 GB RAM, `gmem=40G` (peak host RAM 8 GB, GPU ~14 GB) is comfortable for N=1.

## Results (seed 0, 300 non_spatial_val prompts, 1 image/prompt, 512 px)

| model | steps | Share-CoT | CLIPScore (the default column) | parse failures |
|---|---|---|---|---|
| best COCO-trained student (`docs/CHECKPOINTS.md` default), 4-step | 4 | **0.7727** | 0.3142 | 0/300 |
| SD3.5-M teacher | 28 | **0.7800** | 0.3154 | 0/300 |

Reference points from T2I-CompBench++ TABLE XIII: SD1.4 0.7487, SD2 0.7567, SD3 0.7782, and CTCal
Table 1: SD3 + CTCal 0.787. Both of ours land in that band, and Share-CoT separates student from
teacher by 0.0073 where CLIPScore separates them by 0.0012.

Per-image verdict histograms (the metric is coarse -- five levels, in practice three):

| model | 3 (=0.6) | 4 (=0.8) | 5 (=1.0) |
|---|---|---|---|
| student | 45 | 251 | 4 |
| teacher | 41 | 248 | 11 |

That granularity is the dominant source of noise at N=1: one image moving 4 -> 5 is worth 0.00067 on
the mean. Treat differences below ~0.01 at N=1 as unresolved and run N=10.

## The score directory collides with the CLIPScore one in `compbench_scores/*/scores.json` globs

Its `per_prompt` rows carry `category: "non_spatial"`, same as the CLIPScore file in the same
`compbench_scores/` directory. Anything that globs `compbench_scores/*/scores.json` and keys on
`(category, prompt)` would silently replace the CLIPScore scores with the Share-CoT ones --
non-deterministically, since `glob` returns readdir order -- and arms scored with Share-CoT would be
compared against arms that were not. Every consumer in this repository is guarded
(`evaluator == "sharegpt4v_cot" -> skip`): `eval/compare_arms.py`, `paper/verify_numbers.py`,
`eval/log_to_wandb.py` (logs it as category `non_spatial_sharecot` instead), and
`scripts/eval_alignment.lsf` averages only the eight official columns into `compbench_mean`.
**Anything new that reads these directories must filter on `evaluator` too**, and any further
evaluator must either write outside `compbench_scores/` or never reuse an official category name.
