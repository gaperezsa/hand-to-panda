# Source before every session:  source env.sh
#
# Keeps every cache and temporary file inside the project folder, pins one GPU, and sets
# up headless rendering.

export PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

export TMPDIR="$PROJ/.tmp"
export HF_HOME="$PROJ/.cache/huggingface"
export HF_HUB_CACHE="$PROJ/.cache/huggingface/hub"
export PIP_CACHE_DIR="$PROJ/.cache/pip"
export TORCH_HOME="$PROJ/.cache/torch"
export MPLCONFIGDIR="$PROJ/.cache/matplotlib"
export XDG_CACHE_HOME="$PROJ/.cache"
export WANDB_DIR="$PROJ/outputs/wandb"
mkdir -p "$TMPDIR" "$HF_HUB_CACHE" "$PIP_CACHE_DIR" "$TORCH_HOME" "$MPLCONFIGDIR" "$WANDB_DIR"

# Pin to one GPU (set HUMANOID_GPU to choose which).
# MUJOCO_EGL_DEVICE_ID indexes physical EGL devices, NOT the CUDA_VISIBLE_DEVICES
# remapping, so both are set from the same physical index.
export HUMANOID_GPU="${HUMANOID_GPU:-0}"
export CUDA_VISIBLE_DEVICES="$HUMANOID_GPU"
export MUJOCO_EGL_DEVICE_ID="$HUMANOID_GPU"

# Headless rendering. Falls back to osmesa (CPU) only if EGL is proven broken.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
# A conda env that ships its own Mesa (e.g. anaconda's base) exports this on activation,
# pointing EGL at a vendor directory without the NVIDIA driver. GPU rendering then fails
# with "Cannot initialize a EGL device display". The system default finds NVIDIA.
unset __EGL_VENDOR_LIBRARY_DIRS

# LIBERO reads its config from $LIBERO_CONFIG_PATH/config.yaml, defaulting to ~/.libero.
# If that file is missing, `import libero.libero` calls input() and hangs forever on any
# non-interactive run. Pointing it into the project keeps the build self-contained and
# makes the file something we create deliberately (see tools/setup_libero.py).
export LIBERO_CONFIG_PATH="$PROJ/.cache/libero"
mkdir -p "$LIBERO_CONFIG_PATH"

export SIM_ENV="$PROJ/envs/humanoid-sim"
export VISION_ENV="$PROJ/envs/humanoid-vision"

sim()    { conda activate "$SIM_ENV";    }
vision() { conda activate "$VISION_ENV"; }

echo "env.sh: PROJ=$PROJ  GPU=$HUMANOID_GPU  MUJOCO_GL=$MUJOCO_GL"
echo "        activate with:  conda activate \$SIM_ENV   (or \$VISION_ENV)"
