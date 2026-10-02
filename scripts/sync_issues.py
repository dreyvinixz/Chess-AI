"""Create the six roadmap issues using an existing Git Credential Manager login.

Credentials stay in memory and are never printed or saved by this script.
"""

import base64
import json
import subprocess
import urllib.error
import urllib.request

REPO = "dreyvinixz/Chess-AI"
ISSUES = [
    (
        "Sprint 1: reproducible local CUDA vertical slice",
        "Implement and verify config, board/action encoding, compact network, legal masking, "
        "CUDA doctor, smoke test, CPU CI, and setup docs. Exit: passing tests on CPU and GTX 1650.",
    ),
    (
        "Sprint 2: licensed PGN, Stockfish labels, and resumable training",
        "Stream permitted PGN with game-level splits and fingerprints; produce offline "
        "MultiPV labels; train and resume on GTX 1650; record memory and throughput. "
        "Stockfish must stay out of student search.",
    ),
    (
        "Sprint 3: search and honest local strength evaluation",
        "Compare policy-only and PUCT; add baselines, self-play, local play, and "
        "measured reports with checkpoint/config identifiers and complete W/D/L.",
    ),
    (
        "Sprint 4: assist mode and quality-weighted human experience",
        "Store bot-game observations in SQLite; capture human moves and results; "
        "annotate after games with offline teacher; downweight blunders; support "
        "incremental fine-tuning.",
    ),
    (
        "Sprint 5: safe Chess.com computer-game integration",
        "Inspect current DOM; parse board and game metadata; add multi-signal "
        "fail-closed bot guard, legal execution, autoplay, and a dynamic bot ladder. "
        "Never move in human/live/rated games.",
    ),
    (
        "Sprint 6: documentation, reproducibility, and release quality",
        "Complete English README/docs, dependency and license audit, troubleshooting, "
        "CI, examples, measured reports, optional dashboard, and release checklist.",
    ),
]


def main() -> None:
    credential = subprocess.run(
        ["git", "credential", "fill"],
        input="protocol=https\nhost=github.com\n\n",
        text=True,
        capture_output=True,
        check=True,
    )
    parts = dict(line.split("=", 1) for line in credential.stdout.splitlines() if "=" in line)
    if "password" not in parts:
        raise SystemExit("Git Credential Manager did not return a GitHub credential")
    token = base64.b64encode(
        f"{parts.get('username', 'x-access-token')}:{parts['password']}".encode()
    ).decode()
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Basic {token}",
        "User-Agent": "Chess-AI-roadmap",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    endpoint = f"https://api.github.com/repos/{REPO}/issues"
    try:
        request = urllib.request.Request(endpoint + "?state=all&per_page=100", headers=headers)
        with urllib.request.urlopen(request, timeout=20) as response:
            existing = {issue["title"] for issue in json.load(response)}
        for title, body in ISSUES:
            if title in existing:
                print(f"Already exists: {title}")
                continue
            request = urllib.request.Request(
                endpoint,
                data=json.dumps({"title": title, "body": body}).encode(),
                headers={**headers, "Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=20) as response:
                issue = json.load(response)
            print(f"Created #{issue['number']}: {title}")
    except urllib.error.HTTPError as exc:
        raise SystemExit(
            f"GitHub API returned HTTP {exc.code}; check issue write permission"
        ) from None


if __name__ == "__main__":
    main()
