"""Open the current feature branch as a PR using Git Credential Manager."""

import argparse
import base64
import json
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

REPO = "dreyvinixz/Chess-AI"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--title", required=True)
    parser.add_argument("--body-file", required=True, type=Path)
    parser.add_argument("--base", default="main")
    parser.add_argument("--draft", action="store_true")
    args = parser.parse_args()
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
    payload = {
        "title": args.title,
        "head": branch,
        "base": args.base,
        "body": args.body_file.read_text(encoding="utf-8"),
        "draft": args.draft,
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
    print(f"Created PR #{pr['number']}: {pr['html_url']}")


if __name__ == "__main__":
    main()
