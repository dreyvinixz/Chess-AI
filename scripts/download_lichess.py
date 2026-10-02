"""Download a checksum-verified, CC0 Lichess standard-game monthly archive."""

import argparse
import hashlib
import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE = "https://database.lichess.org/standard"
ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--month", default="2013-01", help="Archive month in YYYY-MM format.")
    parser.add_argument("--allow-large", action="store_true", help="Allow downloads above 1 GiB.")
    args = parser.parse_args()
    if not re.fullmatch(r"20\d{2}-(0[1-9]|1[0-2])", args.month):
        parser.error("month must be YYYY-MM")
    filename = f"lichess_db_standard_rated_{args.month}.pgn.zst"
    with urllib.request.urlopen(f"{BASE}/sha256sums.txt", timeout=30) as response:
        checksums = response.read().decode()
    expected = next(
        (line.split()[0] for line in checksums.splitlines() if line.endswith(filename)), None
    )
    if expected is None:
        parser.error(f"No official checksum for {filename}")
    url = f"{BASE}/{filename}"
    head = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(head, timeout=30) as response:
        size = int(response.headers.get("Content-Length", "0"))
    if size > 1024**3 and not args.allow_large:
        parser.error(f"Archive is {size / 1024**3:.1f} GiB; pass --allow-large to download")
    destination = ROOT / "data" / "raw" / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        with urllib.request.urlopen(url, timeout=120) as response, destination.open("wb") as output:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                output.write(chunk)
    actual = sha256(destination)
    if actual != expected:
        raise ValueError(f"Lichess archive SHA-256 mismatch: expected {expected}, got {actual}")
    manifest = {
        "source": url,
        "file": str(destination),
        "sha256": actual,
        "size_bytes": destination.stat().st_size,
        "license": "CC0-1.0",
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }
    destination.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Verified {destination} ({destination.stat().st_size} bytes, CC0)")


if __name__ == "__main__":
    main()
