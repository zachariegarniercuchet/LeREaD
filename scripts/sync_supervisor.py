"""Sync the article-review tree to the supervisor repo.

Personal main contains everything (outputs, notebooks, raw data, ...).
The supervisor repo must only contain the review subset. A plain
``git push supervisor main`` would leak the excluded files back in,
so this script rebuilds the exact review tree from a source commit in
a scratch worktree and pushes it (fast-forward only, never force).

Usage (from repo root):
    python scripts/sync_supervisor.py [--source HEAD] [--message MSG]
                                      [--dry-run] [--remote supervisor]

Exit codes: 0 = ok (pushed or already up to date), 1 = error,
            2 = nothing to sync (only with --dry-run it still exits 0).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Top-level paths (relative to repo root) taken from the source commit.
ALLOW = [
    "cache",
    "configs",
    "data",
    "img",
    "scripts",
    "src",
    "README.md",
    "Guide_annotation_V2_avril2026_Anonymized_for_review.pdf",
    ".gitignore",
]

# Subtrees that must NOT go to the review repo even though they live
# under an allowed top-level path.
DENY = [
    "cache/chunks/annotated/paragraph",
    "data/annotated/dev_train",
    "data/raw",
    "data/supplementary",
]

# Extra .gitignore lines enforced on the review tree so excluded
# content cannot be re-added by accident.
REVIEW_GITIGNORE_EXTRA = ["output/", "output_coref/", "notebooks/"]

# Filename substrings that must never appear in the review index.
FORBIDDEN_SUBSTRINGS = [
    "notebooks/",
    "output/",
    "output_coref/",
    "AGENTS.md",
    "embeddings.npy",
    "dev_train",
    "data/raw",
    "supplementary",
    "chunks/annotated",
    "__pycache__",
    ".pyc",
]


def run(cmd: list[str], cwd: Path, capture: bool = True) -> str:
    proc = subprocess.run(
        ["git", *cmd],
        cwd=cwd,
        capture_output=capture,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(cmd)} failed:\n{proc.stderr.strip()}")
    return proc.stdout.strip() if capture else ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="HEAD",
                        help="Commit to sync from (default: HEAD)")
    parser.add_argument("--message", default="Sync review tree from personal main",
                        help="Commit message for the review commit")
    parser.add_argument("--remote", default="supervisor",
                        help="Git remote of the review repo (default: supervisor)")
    parser.add_argument("--branch", default="main",
                        help="Branch on the review remote (default: main)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build and verify the tree but do not commit/push")
    args = parser.parse_args()

    root = Path(run(["rev-parse", "--show-toplevel"], Path.cwd())).resolve()
    source = run(["rev-parse", args.source], root)
    print(f"root:   {root}")
    print(f"source: {source}")

    run(["fetch", args.remote], root, capture=False)
    remote_tip = run(["rev-parse", f"{args.remote}/{args.branch}"], root)
    print(f"review tip: {remote_tip}")

    workdir = Path(tempfile.mkdtemp(prefix="leread-review-"))
    try:
        run(["worktree", "add", "--detach", str(workdir),
             f"{args.remote}/{args.branch}"], root)
        run(["rm", "-r", "-q", "."], workdir)
        run(["checkout", source, "--", *ALLOW], workdir)
        for denied in DENY:
            run(["rm", "-r", "-f", "-q", "--cached", "--", denied], workdir)
            shutil.rmtree(workdir / denied, ignore_errors=True)
        # Drop bytecode that may be tracked on either side.
        for path in run(["ls-files"], workdir).splitlines():
            if "__pycache__" in path or path.endswith((".pyc", ".pyo")):
                run(["rm", "-f", "-q", "--cached", "--", path], workdir)
                (workdir / path).unlink(missing_ok=True)

        # Enforce extra .gitignore lines (idempotent, preserves CRLF).
        gi = workdir / ".gitignore"
        raw = gi.read_bytes()
        newline = "\r\n" if b"\r\n" in raw else "\n"
        text = raw.decode().replace("\r\n", "\n")
        missing = [ln for ln in REVIEW_GITIGNORE_EXTRA
                   if f"\n{ln}\n" not in f"\n{text}\n"]
        if missing:
            if not text.endswith("\n"):
                text += "\n"
            text += "".join(f"{ln}\n" for ln in missing)
            gi.write_bytes(text.replace("\n", newline).encode())
            run(["add", ".gitignore"], workdir)

        # Safety net: refuse to commit if anything forbidden is staged.
        bad = [p for p in run(["ls-files", "--cached"], workdir).splitlines()
               if any(s in p.replace("\\", "/") for s in FORBIDDEN_SUBSTRINGS)]
        # 'src/output_control/' files are legitimate sources, not outputs.
        bad = [p for p in bad if not p.replace("\\", "/").startswith("src/output_control/")]
        if bad:
            print("ABORT: forbidden paths in review tree:")
            print("\n".join(f"  {p}" for p in bad[:20]))
            return 1

        status = run(["status", "--porcelain"], workdir)
        print("--- staged stat ---")
        print(run(["diff", "--cached", "--stat"], workdir)[-2000:])
        if not status:
            print("Review tree already up to date, nothing to push.")
            return 0
        if args.dry_run:
            print("Dry run: tree built and verified, no commit/push.")
            return 0

        run(["-c", "user.name=review-sync", "-c",
             "user.email=review-sync@local", "commit", "-q",
             "-m", args.message], workdir)
        new_tip = run(["rev-parse", "HEAD"], workdir)
        print(f"review commit: {new_tip}")
        run(["push", args.remote, f"HEAD:{args.branch}"], workdir,
            capture=False)
        print(f"Pushed {remote_tip[:7]}..{new_tip[:7]} to {args.remote}/{args.branch}")
        return 0
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(workdir)],
                       cwd=root, capture_output=True)
        subprocess.run(["git", "worktree", "prune"], cwd=root,
                       capture_output=True)


if __name__ == "__main__":
    sys.exit(main())
