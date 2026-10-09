#!/usr/bin/env bash
# Fine-tune SmolVLA on one replay dataset. Same recipe for every arm, so the arms differ
# only in their data:  tools/train_arm.sh <arm> <dataset root> <gpu> [seed]
#
# The recipe: SmolVLA architecture with the pretrained SmolVLM backbone, action expert from scratch.
# lerobot/smolvla_base cannot be used directly: it declares three cameras and a 6-dim
# state, LIBERO provides two cameras and an 8-dim state.
set -euo pipefail
ARM=$1; ROOT=$2; GPU=$3; SEED=${4:-0}
source "$(dirname "$0")/../env.sh" >/dev/null
export CUDA_VISIBLE_DEVICES=$GPU
"$SIM_ENV/bin/lerobot-train" \
  --policy.type=smolvla \
  --policy.load_vlm_weights=true \
  --policy.push_to_hub=false \
  --dataset.repo_id="local/$ARM" \
  --dataset.root="$ROOT" \
  --dataset.video_backend=torchcodec \
  --output_dir="$PROJ/outputs/train/$ARM" \
  --steps=15000 \
  --batch_size=64 \
  --save_freq=5000 \
  --log_freq=100 \
  --seed=$SEED \
  --wandb.enable=false
