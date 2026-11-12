#!/usr/bin/env python3
"""
Cross-platform post-push GitHub tag, CI check-run, and release asset verifier
for Devparth7-coder/EVSA (pure Python stdlib, Windows/Linux/macOS).

Usage after pushing `main` and tag `v2.0.2` to GitHub:
    python experiments/verify_github_release.py --tag v2.0.2
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any, Dict


def _fetch_json(url: str) -> Dict[str, Any]:
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "EVSA-Release-Verifier/2.0.2",
        },
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def verify_remote_release(repo: str, tag: str, require_release_assets: bool = False) -> int:
    base = f"https://api.github.com/repos/{repo}"
    print(f"[1/3] Checking Git tag ref `{tag}` on {repo}...")
    try:
        ref_data = _fetch_json(f"{base}/git/ref/tags/{tag}")
    except urllib.error.HTTPError as exc:
        print(f"  FAIL: Tag `{tag}` lookup returned HTTP {exc.code}. Run: git push origin {tag}")
        return 1

    obj = ref_data.get("object", {})
    obj_sha = obj.get("sha", "")
    obj_type = obj.get("type", "")
    commit_sha = obj_sha
    if obj_type == "tag":
        tag_obj = _fetch_json(f"{base}/git/tags/{obj_sha}")
        commit_sha = tag_obj.get("object", {}).get("sha", obj_sha)
    print(f"  OK: Tag `{tag}` ({obj_type}) resolves to commit {commit_sha[:12]}")

    print(f"[2/3] Checking GitHub Actions check-runs for commit {commit_sha[:12]}...")
    checks = _fetch_json(f"{base}/commits/{commit_sha}/check-runs")
    runs = checks.get("check_runs", [])
    if not runs:
        print("  FAIL: Zero check-runs found on commit yet (wait ~60s after pushing for Actions to finish).")
        return 1
    incomplete = [r["name"] for r in runs if r.get("status") != "completed"]
    failed = [r["name"] for r in runs if r.get("status") == "completed" and r.get("conclusion") != "success"]
    for r in runs:
        print(f"  - {r['name']}: status={r.get('status')} conclusion={r.get('conclusion')}")
    if incomplete or failed:
        print("  FAIL: All CI check-runs must be completed with conclusion=success.")
        return 1
    print(f"  OK: All {len(runs)} CI check-runs passed on {commit_sha[:12]}.")

    print(f"[3/3] Checking GitHub Release `{tag}`...")
    try:
        rel = _fetch_json(f"{base}/releases/tags/{tag}")
        assets = [a.get("name", "") for a in rel.get("assets", [])]
        print(f"  OK: Release `{tag}` published ({rel.get('html_url')}); assets={assets}")
    except urllib.error.HTTPError as exc:
        msg = f"Release `{tag}` returned HTTP {exc.code} (not created yet via `gh release create {tag}`)."
        if require_release_assets:
            print(f"  FAIL: {msg}")
            return 1
        print(f"  NOTE: {msg}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify remote GitHub tag, CI check-runs, and release.")
    parser.add_argument("--repo", default="Devparth7-coder/EVSA")
    parser.add_argument("--tag", default="v2.0.2")
    parser.add_argument("--require-release", action="store_true")
    args = parser.parse_args()
    sys.exit(verify_remote_release(args.repo, args.tag, args.require_release))


if __name__ == "__main__":
    main()
