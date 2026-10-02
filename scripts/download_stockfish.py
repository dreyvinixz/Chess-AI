"""Download an official Stockfish release into this repository and verify SHA-256."""

import hashlib
import json
import shutil
import tarfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

TAG = "sf_19"
ASSET = "stockfish-linux-x86-64-universal.tar.gz"
ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "tools" / "stockfish-local"


def request(url: str) -> urllib.request.Request:
    return urllib.request.Request(
        url, headers={"User-Agent": "Chess-AI-setup", "Accept": "application/vnd.github+json"}
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    release_url = f"https://api.github.com/repos/official-stockfish/Stockfish/releases/tags/{TAG}"
    with urllib.request.urlopen(request(release_url), timeout=30) as response:
        release = json.load(response)
    asset = next(item for item in release["assets"] if item["name"] == ASSET)
    expected = asset["digest"].removeprefix("sha256:")
    if len(expected) != 64:
        raise ValueError("Release asset has no usable SHA-256 digest")
    TARGET.mkdir(parents=True, exist_ok=True)
    archive = TARGET / ASSET
    if not archive.exists():
        with (
            urllib.request.urlopen(request(asset["browser_download_url"]), timeout=120) as response,
            archive.open("wb") as output,
        ):
            shutil.copyfileobj(response, output)
    actual = sha256(archive)
    if actual != expected:
        raise ValueError(f"Stockfish SHA-256 mismatch: expected {expected}, got {actual}")
    with tarfile.open(archive, "r:gz") as bundle:
        candidates = [
            member
            for member in bundle.getmembers()
            if member.isfile()
            and Path(member.name).name.startswith("stockfish-")
            and member.mode & 0o111
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"Expected one Stockfish executable in archive, found {len(candidates)}"
            )
        source = bundle.extractfile(candidates[0])
        if source is None:
            raise ValueError("Cannot read Stockfish executable")
        executable = TARGET / "stockfish"
        with source, executable.open("wb") as output:
            shutil.copyfileobj(source, output)
        executable.chmod(0o755)
    manifest = {
        "tag": TAG,
        "asset": ASSET,
        "url": asset["browser_download_url"],
        "sha256": actual,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "license": "GPL-3.0-or-later",
        "executable": str(executable),
    }
    (TARGET / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Stockfish {TAG}: {executable} (SHA-256 verified)")


if __name__ == "__main__":
    main()
