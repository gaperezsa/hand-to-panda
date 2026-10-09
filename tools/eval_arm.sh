#!/usr/bin/env bash
# Evaluate one trained arm on the five libero_spatial tasks the captures cover, with the
# stock init states:  tools/eval_arm.sh <arm> <checkpoint dir> <gpu> [episodes per task]
set -euo pipefail
ARM=$1; CKPT=$2; GPU=$3; N=${4:-20}
source "$(dirname "$0")/../env.sh" >/dev/null
export CUDA_VISIBLE_DEVICES=$GPU MUJOCO_EGL_DEVICE_ID=$GPU
"$SIM_ENV/bin/lerobot-eval" \
  --output_dir="$PROJ/outputs/eval/$ARM" \
  --env.type=libero \
  --env.task=libero_spatial \
  --env.task_ids="[2,0,8,1,6]" \
  --eval.batch_size=1 \
  --eval.n_episodes=$N \
  --env.max_parallel_tasks=1 \
  --policy.path="$CKPT" \
  --policy.n_action_steps=10 \
  --seed=0
