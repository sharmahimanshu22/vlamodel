"""Download everything GPU jobs need, so they can run with no internet access.

Run on the LOGIN node via scripts/prefetch.sh (which sets the cache paths).

    python scripts/prefetch.py                 # LIBERO setup + default eval checkpoint
    python scripts/prefetch.py --models lerobot/smolvla_base
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError

DEFAULT_MODELS = ["lerobot/pi05_libero_finetuned"]

# Policies load only the tokenizer from these base VLM repos; skip the multi-GB weights.
TOKENIZER_REPOS = {
    "pi0": "google/paligemma-3b-pt-224",
    "pi05": "google/paligemma-3b-pt-224",
}
TOKENIZER_FILES = ["*.json", "*.model", "*.txt"]

LIBERO_ASSETS_REPO = "lerobot/libero-assets"


def setup_libero_config() -> Path:
    """Create LIBERO's config.yaml non-interactively and return the libero package dir.

    On first import LIBERO asks on stdin whether to use a custom dataset path;
    answering 'n' keeps the defaults. In an LSF job that prompt would crash on EOF.
    """
    config_file = Path(os.environ["LIBERO_CONFIG_PATH"]) / "config.yaml"
    if not config_file.exists():
        subprocess.run([sys.executable, "-c", "import libero.libero"], input="n\n", text=True, check=True)
    print(f"LIBERO config: {config_file}")

    import libero.libero

    return Path(libero.libero.__file__).parent


def setup_libero_assets(libero_pkg_dir: Path) -> None:
    """Download LIBERO's 3D assets and link them where LIBERO looks first.

    LIBERO checks <package>/assets before falling back to a download into ~/.cache.
    Linking our copy there keeps assets in project storage and jobs offline.
    """
    assets_dir = Path(os.environ["LIBERO_ASSETS_DIR"])
    snapshot_download(repo_id=LIBERO_ASSETS_REPO, repo_type="dataset", local_dir=assets_dir)

    pkg_assets = libero_pkg_dir / "assets"
    if pkg_assets.is_symlink() or not pkg_assets.exists():
        if pkg_assets.is_symlink():
            pkg_assets.unlink()
        pkg_assets.symlink_to(assets_dir, target_is_directory=True)
        print(f"Linked {pkg_assets} -> {assets_dir}")
    else:
        print(f"Package already ships assets at {pkg_assets}; leaving them in place")


def fetch_model(repo_id: str) -> None:
    path = snapshot_download(repo_id=repo_id)
    print(f"{repo_id} -> {path}")

    policy_type = _read_policy_type(Path(path) / "config.json")
    tokenizer_repo = TOKENIZER_REPOS.get(policy_type)
    if tokenizer_repo:
        try:
            snapshot_download(repo_id=tokenizer_repo, allow_patterns=TOKENIZER_FILES)
            print(f"  tokenizer {tokenizer_repo} cached")
        except GatedRepoError:
            sys.exit(
                f"\nERROR: {tokenizer_repo} is gated. Accept its license at "
                f"https://huggingface.co/{tokenizer_repo} (logged in), then run "
                f"`hf auth login` and re-run this script."
            )


def _read_policy_type(config_json: Path) -> str | None:
    import json

    if not config_json.exists():
        return None
    return json.loads(config_json.read_text()).get("type")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS, help="HF model repo ids")
    parser.add_argument("--skip-libero", action="store_true")
    args = parser.parse_args()

    if not args.skip_libero:
        setup_libero_assets(setup_libero_config())
    for repo_id in args.models:
        fetch_model(repo_id)
    print("\nPrefetch complete.")


if __name__ == "__main__":
    main()
