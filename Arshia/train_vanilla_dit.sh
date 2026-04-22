#!/bin/bash
#BSUB -G team361
#BSUB -q training-parallel
#BSUB -n 4
#BSUB -M 128000
#BSUB -R "span[hosts=1] select[mem>128000] rusage[mem=128000]"
#BSUB -gpu "mode=exclusive_process:num=4"
#BSUB -J "dit-vanilla"
#BSUB -o "/nfs/users/nfs_a/ah45/OCTA/logs/dit_vanilla_%J.out"
#BSUB -e "/nfs/users/nfs_a/ah45/OCTA/logs/dit_vanilla_%J.err"
#BSUB -W 72:00

echo "========================================"
echo "Training Vanilla DiT-L/2 on MS-COCO"
echo "No REPA, No ORCA - Baseline"
echo "========================================"

# Setup conda environment via PATH (bypasses activation issues)
export PATH="/software/conda/users/ah45/octa/bin:/lustre/scratch126/cellgen/lotfollahi/ah45/.local/bin:$PATH"
# export PYTHONNOUSERSITE=1
export PYTHONPATH="/nfs/users/nfs_a/ah45/OCTA:$PYTHONPATH"

cd /nfs/users/nfs_a/ah45/OCTA

echo "Python: $(which python)"
echo "Accelerate: $(which accelerate)"
echo "PyTorch version:"
python -c "import torch; print(torch.__version__); print(f'CUDA: {torch.cuda.is_available()}')"

python -m accelerate.commands.launch --num_processes=4 --mixed_precision=fp16 train_t2i.py \
    --proj-coeff=0.0 \
    --enc-type="None" \
    --report-to="wandb" \
    --allow-tf32 \
    --seed=0 \
    --path-type="linear" \
    --prediction="v" \
    --weighting="uniform" \
    --output-dir="/lustre/scratch126/cellgen/lotfollahi/ah45/OCTA/training_checkpoints" \
    --exp-name="dit-l2-vanilla-coco" \
    --data-dir="/lustre/scratch126/cellgen/lotfollahi/ah45/OCTA/data/coco_features"

# Copy best checkpoint to NFS
echo "Copying best checkpoint to NFS..."
mkdir -p /nfs/team361/ah45/OCTA/best_checkpoints
BEST_CKPT=$(ls -t /lustre/scratch126/cellgen/lotfollahi/ah45/OCTA/training_checkpoints/dit-l2-vanilla-coco/checkpoints/*.pt 2>/dev/null | head -1)
if [ -n "$BEST_CKPT" ]; then
    cp "$BEST_CKPT" /nfs/team361/ah45/OCTA/best_checkpoints/dit-l2-vanilla-coco.pt
    echo "✓ Copied: $BEST_CKPT"
else
    echo "! No checkpoint found"
fi

echo "========================================"
echo "Training complete!"
echo "========================================"
