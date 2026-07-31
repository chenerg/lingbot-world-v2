#!/bin/bash
# Two-weather multi-prompt inference with a static camera.
# Usage: bash run_example_07.sh <weights_dir> <num_gpus>
# num_gpus must divide 40 (heads): 2/4/5/8/...

set -euo pipefail
set -x

WEIGHT_DIR=${1:-lingbot-world-v2-14b-causal-fast}
NGPU=${2:-8}

torchrun --nproc_per_node="${NGPU}" generate.py \
  --task i2v-A14B \
  --size '480*832' \
  --ckpt_dir "${WEIGHT_DIR}" \
  --image examples/07/image.jpg \
  --action_path examples/07 \
  --dit_fsdp \
  --t5_fsdp \
  --ulysses_size "${NGPU}" \
  --local_attn_size 18 \
  --sink_size 6 \
  --prompts "The weather turns sunny with a clear blue sky.|||Heavy rain falls under dark storm clouds." \
  --segment_frames 81,81
