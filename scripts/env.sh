# Sourced by every launcher. Adjust to your cluster.
export PATH="${CONDA_BIN:-/software/cellgen/team361/ha11/envs/nichejepa/bin}:$PATH"
export HF_HOME="${HF_HOME:-$PWD/cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$PWD/cache/torch}"
export TOKENIZERS_PARALLELISM=false
export NCCL_NVLS_ENABLE=0
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
# COCO_ROOT must hold annotations/ and train2017/ val2017/
export COCO_ROOT="${COCO_ROOT:-/lustre/scratch126/cellgen/lotfollahi/ha11/COCO}"
# Hugging Face token for the gated SD3.5 weights, if needed
[ -f ~/.secrets/hf_token ] && source ~/.secrets/hf_token
