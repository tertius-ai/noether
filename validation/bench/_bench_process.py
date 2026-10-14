# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Bounded GNU-time measurements that do not orphan subprocess trees."""

from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

TIME = Path("/usr/bin/time")


@dataclass(frozen=True)
class ProcessMeasurement:
    timed_out: bool
    returncode: int
    wall_s: float
    user_s: float
    system_s: float
    peak_rss_kb: int
    output: str

    @property
    def cpu_s(self) -> float:
        return self.user_s + self.system_s


def _children(process: int) -> list[int]:
    try:
        raw = Path(f"/proc/{process}/task/{process}/children").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return []
    return [int(value) for value in raw.split()]


def _descendants(process: int) -> list[int]:
    output: list[int] = []
    pending = _children(process)
    while pending:
        child = pending.pop()
        output.append(child)
        pending.extend(_children(child))
    return output


def _measured_command_pid(timer_pid: int) -> int:
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        children = _children(timer_pid)
        if children:
            return children[0]
        time.sleep(0.001)
    raise RuntimeError("GNU time did not start the measured command")


def _terminate_measurement(process: subprocess.Popen[str]) -> str:
    command_pid = _measured_command_pid(process.pid)
    descendants = _descendants(command_pid)
    for target in reversed(descendants):
        try:
            os.kill(target, signal.SIGTERM)
        except ProcessLookupError:
            pass
    if descendants:
        try:
            output, _ = process.communicate(timeout=2)
            return output
        except subprocess.TimeoutExpired:
            pass
    try:
        os.kill(command_pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        output, _ = process.communicate(timeout=10)
        return output
    except subprocess.TimeoutExpired:
        try:
            os.killpg(command_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        output, _ = process.communicate()
        return output


def timed_process(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str] | None,
    timeout_s: int,
) -> ProcessMeasurement:
    """Measure one command, bounding its whole process group when requested."""
    if not TIME.is_file():
        raise RuntimeError("the benchmark requires GNU /usr/bin/time")
    with tempfile.NamedTemporaryFile(mode="r", encoding="utf-8") as timing:
        started = time.perf_counter()
        process = subprocess.Popen(
            [
                str(TIME),
                "--quiet",
                "-f",
                "%e %U %S %M",
                "-o",
                timing.name,
                "setsid",
                *command,
            ],
            cwd=cwd,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        timed_out = False
        try:
            output, _ = process.communicate(timeout=None if timeout_s == 0 else timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            output = _terminate_measurement(process)
        wall_s = time.perf_counter() - started
        timing.seek(0)
        fields = timing.read().strip().split()
    if len(fields) != 4:
        raise RuntimeError(f"GNU time returned malformed measurements: {fields!r}")
    return ProcessMeasurement(
        timed_out=timed_out,
        returncode=process.returncode,
        wall_s=wall_s,
        user_s=float(fields[1]),
        system_s=float(fields[2]),
        peak_rss_kb=int(fields[3]),
        output=output,
    )
