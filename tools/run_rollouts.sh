#!/usr/bin/env bash
# N workers over one rollout job list, spread over the four GPUs (policy and rendering):
#   tools/run_rollouts.sh <n_workers> <name> <policy_rollout args...>
set -euo pipefail
N=$1; NAME=$2; shift 2
source "$(dirname "$0")/../env.sh" >/dev/null
mkdir -p "$PROJ/outputs/rollouts"
# without a cap every worker starts one thread per core and they starve each other
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
for i in $(seq 0 $((N-1))); do
  GPU=$((i % 4))
  CUDA_VISIBLE_DEVICES=$GPU MUJOCO_EGL_DEVICE_ID=$GPU \
    setsid nohup "$SIM_ENV/bin/python" -u "$PROJ/tools/policy_rollout.py" "$@" --worker $i/$N \
    > "$PROJ/outputs/rollouts/${NAME}_w$i.log" 2>&1 < /dev/null &
done
echo "$N workers launched for $NAME"
