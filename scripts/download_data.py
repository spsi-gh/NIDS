"""Download pinned public copies of the official UNSW-NB15 partitions."""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "raw"
REVISION = "6f5f54594dfc80c84264aec4c7cc3d9b162f1e2f"
BASE = f"https://huggingface.co/datasets/Mouwiya/UNSW-NB15-small/resolve/{REVISION}"
FILES = {
    "UNSW_NB15_training-set.csv": "bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa",
    "UNSW_NB15_testing-set.csv": "734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        target = DEST / name
        if target.exists() and sha256(target) != expected:
            raise ValueError(
                f"{name} already exists with a different hash. "
                "Keep it if it came directly from UNSW and run preprocessing, "
                "or move it aside before downloading this pinned copy."
            )
        if not target.exists():
            print(f"Downloading {name}...")
            temporary = target.with_suffix(".download")
            urllib.request.urlretrieve(f"{BASE}/{name}", temporary)
            if sha256(temporary) != expected:
                temporary.unlink(missing_ok=True)
                raise ValueError(f"Checksum mismatch for {name}")
            temporary.replace(target)
        print(f"Verified {name}: {sha256(target)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
