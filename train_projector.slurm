#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=60GB
#SBATCH -A punim2198
#SBATCH -o slurm-projector-%j.out

module load GCCcore/11.3.0
module load Python/3.11.3

export HF_HOME=cache/huggingface
export TORCH_HOME=cache/torch
export XDG_CACHE_HOME=cache/xdg

source virtualenv/Python3.11.3/bin/activate

# Install deps on first run
pip install -q -r requirements.txt

# Train projector: align DiT hidden states with DINOv2 features (COCO captions)
python train_projector.py \
  --coco_path /data/gpfs/datasets/COCO \
  --num_steps 5000 \
  --batch_size 4 \
  --lr 1e-4 \
  --num_inference_steps 20 \
  --height 512 \
  --width 512 \
  --output_dir checkpoints/projector \
  --save_every 500 \
  --log_every 50

my-job-stats -a -n -s
