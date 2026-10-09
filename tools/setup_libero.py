#!/usr/bin/env python3
"""Make LIBERO importable without it hanging, and fetch the assets it ships without.

Two separate problems:

1. `libero/libero/__init__.py` calls input() at import time when its config.yaml is
   missing -- it asks whether you want a custom dataset path. On a headless, non-
   interactive run that blocks forever with no output. So the config file has to exist
   *before* the first import. $LIBERO_CONFIG_PATH redirects it into the project.

2. `bddl_files` (135) and `init_files` (250) ship inside the wheel, but `assets` does
   not. Left alone, libero tries to pull them from the Hub lazily at env-construction
   time, which is slow and fails behind a flaky network. Fetch them up front.

    python tools/setup_libero.py
"""
import os
from pathlib import Path

import yaml


def main() -> None:
    cfg_dir = Path(os.environ.get("LIBERO_CONFIG_PATH", Path.home() / ".libero"))
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_file = cfg_dir / "config.yaml"

    # Resolve the package WITHOUT importing libero.libero, which is what hangs.
    import importlib.util
    spec = importlib.util.find_spec("libero")
    pkg = Path(spec.origin).parent / "libero"

    assets = Path(os.environ["PROJ"]) / "data" / "libero" / "assets"
    if not (assets / "stable_scanned_objects").is_dir():
        from huggingface_hub import snapshot_download
        print(f"downloading lerobot/libero-assets -> {assets}")
        snapshot_download(repo_id="lerobot/libero-assets", repo_type="dataset",
                          local_dir=str(assets))
    else:
        print(f"assets already present at {assets}")

    cfg = {
        "assets": str(assets),
        "bddl_files": str(pkg / "bddl_files"),
        "datasets": str(Path(os.environ["PROJ"]) / "data" / "libero" / "datasets"),
        "init_states": str(pkg / "init_files"),
        "benchmark_root": str(pkg),
    }
    # The config.yaml "assets" key is only read by get_libero_path("assets"). The newer
    # get_assets_path() ignores it entirely: it looks for <package>/libero/assets and,
    # failing that, re-downloads to ~/.cache/libero regardless of what we configured.
    # Symlinking the package path at our copy satisfies both and stops the second
    # 422 MB download (which otherwise fires at env-construction time, ~70 s in).
    pkg_assets = pkg / "assets"
    if not pkg_assets.exists():
        pkg_assets.symlink_to(assets, target_is_directory=True)
        print(f"symlinked {pkg_assets} -> {assets}")
    elif pkg_assets.is_symlink():
        print(f"symlink already present: {pkg_assets} -> {pkg_assets.resolve()}")

    cfg_file.write_text(yaml.safe_dump(cfg, sort_keys=True))
    print(f"wrote {cfg_file}")
    for k, v in sorted(cfg.items()):
        print(f"  {k:14s} {v}  {'OK' if Path(v).exists() else 'MISSING'}")


if __name__ == "__main__":
    main()
