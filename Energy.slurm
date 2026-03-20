#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=60GB
#SBATCH -A punim2198 

module load GCCcore/11.3.0
module load Python/3.11.3

export HF_HOME=cache/huggingface
export TORCH_HOME=cache/torch
export XDG_CACHE_HOME=cache/xdg

source virtualenv/Python3.11.3/bin/activate

# Install dependencies if needed (uncomment for first run):
# pip install -r requirements.txt

# Train projector (Step 1 of Energy-based methodology):
# python train_projector.py --num_steps 5000 --batch_size 4 --output_dir checkpoints/projector

jupyter notebook --no-browser --port=8889 --ip=0.0.0.0


my-job-stats -a -n -s