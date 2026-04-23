bsub -J dit-xl-coco -G team361 -q training-parallel \
  -n 32 -W 72:00 \
  -R "span[hosts=1]" -R "select[mem>512000]" -R "rusage[mem=512000]" -M 512000 \
  -gpu "num=4:mode=exclusive_process" \
  -cwd /lustre/scratch126/cellgen/lotfollahi/ha11/EnergyVLM/Arshia \
  -o /lustre/scratch126/cellgen/lotfollahi/ha11/EnergyVLM/logs/lsf-dit-xl-coco-%J.out \
  -e /lustre/scratch126/cellgen/lotfollahi/ha11/EnergyVLM/logs/lsf-dit-xl-coco-%J.err \
  /bin/bash -c '
    export HF_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/EnergyVLM/cache/huggingface
    export TORCH_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/EnergyVLM/cache/torch
    export XDG_CACHE_HOME=/lustre/scratch126/cellgen/lotfollahi/ha11/EnergyVLM/cache/xdg
    source ~/.secrets/hf_token
    export OMP_NUM_THREADS=1
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    export TOKENIZERS_PARALLELISM=false
    export PYTHONPATH=/lustre/scratch126/cellgen/lotfollahi/ha11/EnergyVLM/Arshia:$PYTHONPATH
    mkdir -p /lustre/scratch126/cellgen/lotfollahi/ha11/EnergyVLM/logs
    python -m accelerate.commands.launch --num_processes=4 --mixed_precision=bf16 train_t2i.py \
      --depth=28 \
      --hidden-size=1152 \
      --num-heads=16 \
      --proj-coeff=0.0 \
      --enc-type=None \
      --report-to=wandb \
      --allow-tf32 \
      --seed=0 \
      --path-type=linear \
      --prediction=v \
      --weighting=uniform \
      --batch-size=128 \
      --max-train-steps=400000 \
      --checkpointing-steps=50000 \
      --sampling-steps=2000 \
      --learning-rate=1e-4 \
      --num-workers=4 \
      --output-dir=/lustre/scratch126/cellgen/lotfollahi/ha11/EnergyVLM/training_checkpoints \
      --exp-name=dit-xl2-vanilla-coco \
      --data-dir=/lustre/scratch126/cellgen/lotfollahi/ah45/OCTA/data/coco_features
  '
