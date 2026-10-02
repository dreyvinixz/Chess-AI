"""Open the current feature branch as a draft PR using Git Credential Manager."""

import base64
import json
import subprocess
import urllib.error
import urllib.request

REPO = "dreyvinixz/Chess-AI"


def main() -> None:
    credentials = subprocess.run(
        ["git", "credential", "fill"],
        input="protocol=https\nhost=github.com\n\n",
        text=True,
        capture_output=True,
        check=True,
    )
    parts = dict(line.split("=", 1) for line in credentials.stdout.splitlines() if "=" in line)
    if "password" not in parts:
        raise SystemExit("Git Credential Manager did not return a GitHub credential")
    token = base64.b64encode(
        f"{parts.get('username', 'x-access-token')}:{parts['password']}".encode()
    ).decode()
    branch = subprocess.run(
        ["git", "branch", "--show-current"], capture_output=True, text=True, check=True
    ).stdout.strip()
    if branch == "main":
        raise SystemExit("Create a feature branch before opening a PR")
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Basic {token}",
        "Content-Type": "application/json",
        "User-Agent": "Chess-AI-roadmap",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    body = """## Scope
- Package and configure the compact chess student for WSL2 GTX 1650.
- Add legal actions, policy/value network, PGN preparation, offline teacher labels,
  training, PUCT, local play, and CLI.
- Add setup/design documentation, CPU CI, and sprint roadmap.

## Evidence
- WSL2 doctor: GTX 1650, 4096 MiB, PyTorch 2.6.0+cu124, FP16 operation passed.
- `ruff check src tests`: passed.
- `pytest -q`: 12 passed.
- `chess-ai smoke-test`: passed on CUDA.
- `python -m build`: sdist and wheel built.

## Limits
- Stockfish is absent locally; teacher labeling still needs a real UCI run.
- Chess.com browser integration and bot results belong to later sprints.

Closes #1 after CI verification.
"""
    payload = {
        "title": "feat: establish local CUDA chess vertical slice",
        "head": branch,
        "base": "main",
        "body": body,
        "draft": True,
    }
    request = urllib.request.Request(
        f"https://api.github.com/repos/{REPO}/pulls",
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            pr = json.load(response)
    except urllib.error.HTTPError as exc:
        raise SystemExit(
            f"GitHub API returned HTTP {exc.code}; check PR write permission"
        ) from None
    print(f"Created draft PR #{pr['number']}: {pr['html_url']}")


if __name__ == "__main__":
    main()
