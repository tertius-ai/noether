#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Summarize bench_threads CSV as Markdown without assuming a host or thread grid."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Measurement:
    n: int
    threads: int
    serial: float
    parallel: float
    libm4ri: float


def read_measurements(path: Path) -> list[Measurement]:
    measurements = []
    seen = set()
    with path.open(newline="", encoding="utf-8") as source:
        for line_number, row in enumerate(csv.reader(source), 1):
            if not row or row[0].lstrip().startswith("#"):
                continue
            try:
                n, threads, serial, parallel, libm4ri = row
                measurement = Measurement(
                    int(n), int(threads), float(serial), float(parallel), float(libm4ri)
                )
                if measurement.n <= 0 or measurement.threads <= 0:
                    raise ValueError("N and threads must be positive")
                for value in (measurement.parallel, measurement.libm4ri):
                    if not math.isfinite(value) or value <= 0:
                        raise ValueError("parallel and libm4ri timings must be finite and positive")
                if not math.isnan(measurement.serial) and (
                    not math.isfinite(measurement.serial) or measurement.serial <= 0
                ):
                    raise ValueError("serial timing must be positive or NaN when skipped")
                key = measurement.n, measurement.threads
                if key in seen:
                    raise ValueError("duplicate (N, threads) measurement")
                seen.add(key)
                measurements.append(measurement)
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    if not measurements:
        raise ValueError(f"{path}: no measurements")
    return measurements


def summarize(measurements: list[Measurement]) -> None:
    print("# m4rst vs libm4ri — raw eight-table kernel thread sweep\n")
    print("Times are milliseconds; m4rst / libm4ri below one favors m4rst.\n")
    print("Host and affinity provenance are recorded in the runner's companion log.\n")
    sizes = sorted({measurement.n for measurement in measurements})
    for n in sizes:
        rows = sorted(
            (measurement for measurement in measurements if measurement.n == n),
            key=lambda measurement: measurement.threads,
        )
        print(f"## N = {n}\n")
        print("| threads | m4rst parallel ms | libm4ri ms | m4rst / libm4ri |")
        print("|---:|---:|---:|---:|")
        for row in rows:
            print(
                f"| {row.threads} | {row.parallel:.6f} | {row.libm4ri:.6f} | "
                f"{row.parallel / row.libm4ri:.2f}× |"
            )
        for row in rows:
            if math.isfinite(row.serial):
                print(
                    f"\nSerial eight-table baseline: {row.serial:.6f} ms "
                    f"(measured in the T={row.threads} process)."
                )
        print()

    print("## Best measured parallel time for each library\n")
    print("| N | m4rst ms (threads) | libm4ri ms (threads) | libm4ri / m4rst |")
    print("|---:|---:|---:|---:|")
    for n in sizes:
        rows = [measurement for measurement in measurements if measurement.n == n]
        best_m = min(rows, key=lambda row: row.parallel)
        best_l = min(rows, key=lambda row: row.libm4ri)
        print(
            f"| {n} | {best_m.parallel:.6f} (T={best_m.threads}) | "
            f"{best_l.libm4ri:.6f} (T={best_l.threads}) | "
            f"{best_l.libm4ri / best_m.parallel:.2f}× |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, nargs="?", default=Path("bench-runs/m4rst/threads.csv"))
    args = parser.parse_args()
    try:
        measurements = read_measurements(args.csv)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    summarize(measurements)


if __name__ == "__main__":
    main()
