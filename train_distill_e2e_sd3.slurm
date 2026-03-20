#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=80GB
#SBATCH -A punim2198

module load GCCcore/11.3.0
module load Python/3.11.3

export HF_HOME=cache/huggingface
export TORCH_HOME=cache/torch
export XDG_CACHE_HOME=cache/xdg

source virtualenv/Python3.11.3/bin/activate

pip install -q -r requirements.txt

# End-to-end energy-based distillation for SD3.5
# Teacher: stabilityai/stable-diffusion-3.5-large (8.1B)
# Student: stabilityai/stable-diffusion-3.5-medium (2.5B)
python train_distill_e2e_sd3.py \
  --height 512 \
  --width 512 \
  --coco_path /data/gpfs/datasets/COCO \
  --num_steps 10000 \
  --K 4 \
  --tau 0.1 \
  --lambda_align 0 \
  --beta_kl 1 \
  --lr_student 1e-5 \
  --lr_projector 1e-4 \
  --num_inference_steps 28 \
  --output_dir checkpoints/distill_e2e_sd3_KL_1 \
  --save_every 1000 \
  --log_every 50

my-job-stats -a -n -s
