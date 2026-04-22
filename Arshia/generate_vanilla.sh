#!/bin/bash
#BSUB -G team361
#BSUB -q training-parallel
#BSUB -n 4
#BSUB -M 128000
#BSUB -R "span[hosts=1] select[mem>128000] rusage[mem=128000]"
#BSUB -gpu "mode=exclusive_process:num=4"
#BSUB -J "dit-gen-vanilla"
#BSUB -o "/nfs/users/nfs_a/ah45/OCTA/logs/dit_gen_vanilla_%J.out"
#BSUB -e "/nfs/users/nfs_a/ah45/OCTA/logs/dit_gen_vanilla_%J.err"
#BSUB -W 24:00

echo "========================================"
echo "Generating 30K samples from VANILLA model"
echo "========================================"

# Setup conda environment via PATH
export PATH="/software/conda/users/ah45/octa/bin:/lustre/scratch126/cellgen/lotfollahi/ah45/.local/bin:$PATH"
export PYTHONPATH="/nfs/users/nfs_a/ah45/OCTA:$PYTHONPATH"

cd /nfs/users/nfs_a/ah45/OCTA

echo "Python: $(which python)"
echo "PyTorch version:"
python -c "import torch; print(torch.__version__); print(f'CUDA: {torch.cuda.is_available()}')"

# Checkpoint path
VANILLA_CKPT="/lustre/scratch126/cellgen/lotfollahi/ah45/OCTA/training_checkpoints/dit-l2-vanilla-coco/checkpoints/0400000.pt"

# Sample directory
SAMPLE_DIR="/lustre/scratch126/cellgen/lotfollahi/ah45/OCTA/samples"
mkdir -p "$SAMPLE_DIR"

# Common args
COMMON_ARGS="
    --data-dir=/lustre/scratch126/cellgen/lotfollahi/ah45/OCTA/data/coco_features
    --num-samples=30080
    --resolution=256
    --vae=ema
    --per-proc-batch-size=32
    --mode=ode
    --cfg-scale=1.0
    --path-type=linear
    --num-steps=50
    --heun
    --global-seed=42
"

echo ""
echo "Starting VANILLA generation..."
echo "Checkpoint: $VANILLA_CKPT"
echo ""

python -m accelerate.commands.launch --num_processes=4 generate_t2i.py \
    --ckpt="$VANILLA_CKPT" \
    --sample-dir="$SAMPLE_DIR" \
    --prefix="vanilla" \
    --projector-embed-dims="0" \
    --encoder-depth=0 \
    $COMMON_ARGS

echo ""
echo "========================================"
echo "VANILLA generation complete!"
echo "========================================"
echo "Output: $SAMPLE_DIR/vanilla-*.npz"
echo "========================================"