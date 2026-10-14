#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Record capabilities for an exact Noether checkout on an HPC site.

This bootstrap probe deliberately supports Python 3.6 because long-lived HPC
login-node installations may be older than Noether's Python runtime floor.
"""

import argparse
import getpass
import json
import os
import platform
import shlex
import shutil
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SITE_SCHEMA = "noether.slurm-site.v1"
PROBE_COMMANDS = (
    "bash",
    "git",
    "python3",
    "cargo",
    "rustc",
    "gcc",
    "clang",
    "mpicc",
    "mpirun",
    "srun",
    "sinfo",
    "sbatch",
    "squeue",
    "sacct",
    "sacctmgr",
    "apptainer",
)


class ProbeError(RuntimeError):
    pass


def _command_text(command):
    return " ".join(shlex.quote(part) for part in command)


def _run(command):
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=False,
        )
    except OSError as error:
        raise ProbeError("could not run {!r}: {}".format(command[0], error))
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        suffix = ": {}".format(detail) if detail else ""
        raise ProbeError(
            "command failed ({}): {}{}".format(completed.returncode, _command_text(command), suffix)
        )
    return completed.stdout.strip()


def _capture(command, timeout=20.0):
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            start_new_session=True,
        )
    except OSError as error:
        return {
            "command": _command_text(command),
            "returncode": 127,
            "stdout": "",
            "stderr": str(error),
        }
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return {
            "command": _command_text(command),
            "returncode": process.returncode,
            "stdout": stdout.rstrip(),
            "stderr": stderr.rstrip(),
        }
    except subprocess.TimeoutExpired as error:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        return {
            "command": _command_text(command),
            "returncode": 124,
            "stdout": stdout.rstrip(),
            "stderr": (stderr.rstrip() + "\n" + str(error)).strip(),
        }


def _module_probe(name):
    script = (
        "if type module >/dev/null 2>&1; then "
        "module -r spider {}; "
        "else echo 'environment-modules command is unavailable' >&2; exit 127; fi"
    ).format(shlex.quote(name))
    return _capture(("bash", "-lc", script))


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--host-label", required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    source = args.source.resolve()
    revision = _run(("git", "-C", str(source), "rev-parse", "HEAD"))
    if revision != args.expected_revision:
        raise ProbeError("source is {}, expected {}".format(revision, args.expected_revision))
    if _run(("git", "-C", str(source), "status", "--porcelain")):
        raise ProbeError("source tree is dirty: {}".format(source))

    commands = {name: shutil.which(name) for name in PROBE_COMMANDS}
    if commands["sinfo"]:
        partitions = _capture(("sinfo", "-h", "-o", "%P|%a|%D|%c|%m|%G|%l"))
    else:
        partitions = _capture(("sinfo", "--version"))
    if commands["sacctmgr"]:
        associations = _capture(
            (
                "sacctmgr",
                "-n",
                "-P",
                "show",
                "assoc",
                "user={}".format(getpass.getuser()),
                "format=Account,Partition,QOS,MaxJobs,MaxWall",
            )
        )
    else:
        associations = _capture(("sacctmgr", "--version"))
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except AttributeError:
        affinity = []

    payload = {
        "schema": SITE_SCHEMA,
        "probed_at": datetime.now(timezone.utc).isoformat(),
        "host": args.host_label,
        "remote_root": args.remote_root,
        "source": {"path": str(source), "revision": revision, "clean": True},
        "system": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "visible_cpus": os.cpu_count(),
            "affinity_cpus": affinity,
        },
        "commands": commands,
        "details": {
            "lscpu": _capture(("lscpu",)),
            "filesystem": _capture(("df", "-P", str(source))),
            "mpi_version": _capture(("mpirun", "--version")),
            "rust_version": _capture(("rustc", "--version")),
            "cargo_version": _capture(("cargo", "--version")),
        },
        "modules": {name: _module_probe(name) for name in ("mpi", "gcc", "clang", "rust")},
        "slurm": {"partitions": partitions, "associations": associations},
    }
    _write_json(args.output.resolve(), payload)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ProbeError as error:
        print("SLURM SITE PROBE FAILED: {}".format(error), file=sys.stderr)
        raise SystemExit(1)
