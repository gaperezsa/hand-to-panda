#!/usr/bin/env bash
# Build the humanoid-sim environment. Idempotent enough to re-run after a failure.
#
# Two deliberate choices recorded here rather than in prose:
#   - python 3.12 because lerobot declares requires-python >=3.12
#   - ffmpeg from conda-forge because torchcodec needs a real ffmpeg, not a pip shim
set -euo pipefail
source "$(dirname "$0")/../env.sh" >/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"

LOG="$PROJ/outputs/setup_sim_env.log"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "===== $(date -Is) setup_sim_env.sh ====="

echo "--- df / before ---"; df -h / | tail -1

if [ ! -d "$SIM_ENV" ]; then
  echo "--- conda create ---"
  conda create --prefix "$SIM_ENV" python=3.12 -y -q
fi
conda activate "$SIM_ENV"
echo "python: $(which python) $(python -V 2>&1)"

echo "--- ffmpeg + cmake from conda-forge ---"
# cmake is a BUILD dependency, not a runtime one: hf-libero -> hf-egl-probe (and
# egl_probe) compile a C extension that probes EGL devices, and their wheel build
# dies with "RuntimeError: CMake must be installed." if cmake is absent.
conda install --prefix "$SIM_ENV" -c conda-forge ffmpeg cmake -y -q

echo "--- clone lerobot ---"
mkdir -p "$PROJ/third_party"
if [ ! -d "$PROJ/third_party/lerobot/.git" ]; then
  git clone --quiet https://github.com/huggingface/lerobot.git "$PROJ/third_party/lerobot"
fi
cd "$PROJ/third_party/lerobot"
git checkout --quiet 8c920c4270460851cedd2737657584586d3dc66f      # the commit everything here was run with
echo "LEROBOT_COMMIT=$(git rev-parse HEAD)"
echo "LEROBOT_DESCRIBE=$(git describe --tags --always)"
echo "LEROBOT_DATE=$(git log -1 --format=%cI)"
python -c "import tomllib,sys; print('LEROBOT_VERSION='+tomllib.load(open('pyproject.toml','rb'))['project']['version'])"

echo "--- pip install -e .[all]  (guard rail: fall back to targeted extras on failure) ---"
if ! pip install -e ".[all]"; then
  echo "!!! [all] FAILED -- falling back to the extras this project needs"
  pip install -e ".[libero,smolvla,pi,training]"
  echo "FALLBACK_USED=1"
fi

echo "--- df / after ---"; df -h / | tail -1
echo "--- env size ---"; du -sh "$SIM_ENV"
echo "===== $(date -Is) done ====="
