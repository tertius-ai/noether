#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Select a source commit for documentation and guard the stable deployment."""

from __future__ import annotations

import argparse
import os
import re
import subprocess

STABLE_TAG = re.compile(r"v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\Z")


def latest_stable(tags: list[str]) -> str:
    versions = {
        tuple(int(part) for part in match.groups()): tag
        for tag in tags
        if (match := STABLE_TAG.fullmatch(tag))
    }
    if not versions:
        raise ValueError("no stable vMAJOR.MINOR.PATCH tag exists")
    return versions[max(versions)]


def select_source(tags: list[str], event: str, ref: str, requested: str = "") -> tuple[str, bool]:
    """Branch/PR builds check HEAD; only the highest stable tag is publishable."""
    if event == "workflow_dispatch":
        latest = latest_stable(tags)
        tag = requested or latest
        if tag != latest:
            raise ValueError(f"only the latest stable tag can be published: {latest}")
        return tag, True
    if event == "push" and ref.startswith("refs/tags/"):
        tag = ref.removeprefix("refs/tags/")
        if STABLE_TAG.fullmatch(tag):
            return tag, tag == latest_stable(tags)
    return "HEAD", False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-latest", help="reject a stale deployment immediately before publishing"
    )
    args = parser.parse_args()
    tags = subprocess.check_output(["git", "tag", "--list"], text=True).splitlines()
    if args.verify_latest:
        latest = latest_stable(tags)
        if args.verify_latest != latest:
            raise SystemExit(
                f"refusing stale documentation deployment: latest stable tag is {latest}"
            )
        return 0
    try:
        source, publish = select_source(
            tags,
            os.environ["GITHUB_EVENT_NAME"],
            os.environ["GITHUB_REF"],
            os.environ.get("DOCS_TAG", ""),
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{source}^{{commit}}"], text=True
    ).strip()
    print(f"commit={commit}")
    print(f"tag={source if source != 'HEAD' else ''}")
    print(f"publish={str(publish).lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
