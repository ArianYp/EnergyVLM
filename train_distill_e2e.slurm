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

# End-to-end energy-based distillation (no pre-phase projector training)
# Student = smaller DiT (14 layers) from teacher. PixArt-XL-2-256x256 has no model_index.json.
python train_distill_e2e.py \
  --student_layers 14 \
  --height 512 \
  --width 512 \
  --coco_path /data/gpfs/datasets/COCO \
  --num_steps 10000 \
  --K 4 \
  --tau 0.1 \
  --lambda_align 0 \
  --beta_kl 0.1 \
  --lr_student 1e-5 \
  --lr_projector 1e-4 \
  --num_inference_steps 20 \
  --output_dir checkpoints/distill_e2e \
  --save_every 500 \
  --log_every 50

my-job-stats -a -n -s
