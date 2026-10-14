# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""GAP subprocess driver.

`run_gap(script)` spawns `gap` in quiet batch mode, feeds the script on
stdin, and returns stdout. Scripts emit results with explicit
`Print("MARKER: ", value, "\\n");` lines that the per-domain modules
parse — GAP in this mode does not echo statement values, so marked
lines are the contract.
"""

from __future__ import annotations

import shutil
import subprocess

# `-q`            quiet — no banner
# `-A`            no autoloaded packages — faster, deterministic startup
# `--quitonbreak` on an error, quit (nonzero exit) instead of dropping
#                 into the interactive break loop, which would hang a
#                 piped subprocess.
_GAP_ARGV = ["gap", "-q", "-A", "--quitonbreak"]
_DEFAULT_TIMEOUT = 120


class GapError(RuntimeError):
    """GAP failed: not installed, timed out, or exited nonzero."""


def gap_available() -> bool:
    """True iff the `gap` binary is on PATH."""
    return shutil.which("gap") is not None


def run_gap(script: str, timeout: int = _DEFAULT_TIMEOUT) -> str:
    """Run `script` in GAP batch mode; return stdout.

    The script should end with `QUIT;`. Raises `GapError` on any
    failure.
    """
    try:
        result = subprocess.run(
            _GAP_ARGV,
            input=script,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as e:
        raise GapError("gap is not installed / not on PATH") from e
    except subprocess.TimeoutExpired as e:
        raise GapError(f"gap timed out after {timeout}s") from e

    if result.returncode != 0:
        raise GapError(
            f"gap exited {result.returncode}:\n"
            f"--- stderr ---\n{result.stderr}\n"
            f"--- stdout ---\n{result.stdout}"
        )
    return result.stdout
