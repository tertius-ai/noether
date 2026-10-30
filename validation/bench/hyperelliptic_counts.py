#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Paired full-Frobenius computations through the public Rust facade.

Build ``cargo build --release -p noether --example bench_hyperelliptic_counts``
and run this driver through ``scripts/bench-pin.sh``. Each worker constructs
the same deterministic dense smooth model outside timing, warms the operation,
then records fresh complete calls. Timings include result allocation and
evidence construction, excluding fixture generation, formatting, and output
destruction. The default goal returns the complete H1 Frobenius polynomial:
direct enumeration computes N1 through Ng and reconstructs it in Rust;
Kedlaya computes it p-adically. ``--goal point_count`` separately measures the
single-count workflow. All returned coefficients must agree. There is no
mathematical implementation in this orchestration script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import queue
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from _bench_cpu import cpu_provenance  # type: ignore[import-not-found]

ROOT = Path(__file__).resolve().parents[2]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary", type=Path, default=ROOT / "target/release/examples/bench_hyperelliptic_counts"
    )
    parser.add_argument(
        "--baseline-binary",
        type=Path,
        help="add paired Kedlaya calls from a previously built executable",
    )
    parser.add_argument("--baseline-source-sha256", default=None)
    parser.add_argument(
        "--corpus", type=Path, help="JSON model manifest with explicit fixtures and methods"
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        default=["3:1", "3:2"],
        help="prime:residue-degree presentations",
    )
    parser.add_argument("--genera", nargs="+", type=int, default=list(range(1, 7)))
    parser.add_argument(
        "--cases",
        nargs="+",
        help="explicit prime:residue-degree:genus cases, replacing the field/genus product",
    )
    parser.add_argument("--goal", choices=["frobenius", "point_count"], default="frobenius")
    parser.add_argument("--parities", nargs="+", choices=["odd", "even"], default=["odd", "even"])
    parser.add_argument(
        "--methods", nargs="+", choices=["direct", "kedlaya"], default=["direct", "kedlaya"]
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument(
        "--timeout", type=float, default=30, help="maximum seconds without progress from one call"
    )
    parser.add_argument("--gp", type=Path, help="optional validation-only PARI/GP executable")
    parser.add_argument("--revision", default="working-tree")
    parser.add_argument("--source-sha256", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parsed = parser.parse_args()
    if parsed.repeats < 1 or parsed.warmups < 0 or parsed.timeout <= 0:
        parser.error("repeats and timeout must be positive; warmups must be nonnegative")
    if parsed.corpus and (parsed.cases or parsed.baseline_binary):
        parser.error("--corpus cannot be combined with --cases or --baseline-binary")
    if parsed.cases:
        try:
            if any(
                len(case.split(":")) != 3 or any(int(v) <= 0 for v in case.split(":"))
                for case in parsed.cases
            ):
                parser.error("--cases requires positive prime:residue-degree:genus triples")
        except ValueError:
            parser.error("--cases requires integer prime:residue-degree:genus triples")
    return parsed


def selected_models(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.corpus:
        document = json.loads(args.corpus.read_text())
        if document.get("schema") != 1 or not isinstance(document.get("models"), list):
            raise ValueError("the corpus requires schema 1 and a models list")
        models = document["models"]
    else:
        models = []
        for specification in args.cases or args.fields:
            values = list(map(int, specification.split(":")))
            p, n = values[:2]
            for genus in values[2:] if args.cases else args.genera:
                for parity in args.parities:
                    models.append(
                        dict(
                            id=f"{p}:{n}:{genus}:{parity}",
                            prime=p,
                            residue_degree=n,
                            genus=genus,
                            parity=parity,
                        )
                    )
    identifiers: set[str] = set()
    selected = []
    for model in models:
        if not isinstance(model, dict) or set(model) - {
            "id",
            "prime",
            "residue_degree",
            "genus",
            "parity",
            "fixture",
            "methods",
        }:
            raise ValueError("invalid corpus model keys")
        if not isinstance(model.get("id"), str) or model["id"] in identifiers:
            raise ValueError("model identifiers must be distinct strings")
        identifiers.add(model["id"])
        if any(
            type(model.get(k)) is not int or model[k] <= 0
            for k in ["prime", "residue_degree", "genus"]
        ):
            raise ValueError("model field and genus parameters must be positive integers")
        if model["parity"] not in {"odd", "even"} or model["genus"] > 64:
            raise ValueError("invalid model parity or genus")
        fixture = model.get("fixture")
        if fixture is not None and not isinstance(fixture, dict):
            raise ValueError("a fixture must be an object")
        methods = model.get("methods", args.methods)
        if (
            not isinstance(methods, list)
            or not methods
            or len(set(methods)) != len(methods)
            or any(method not in {"direct", "kedlaya"} for method in methods)
        ):
            raise ValueError("model methods must be distinct direct/kedlaya choices")
        methods = [method for method in methods if method in args.methods]
        if methods and model["parity"] in args.parities:
            selected.append({**model, "methods": methods})
    if not selected:
        raise ValueError("the selectors admitted no models")
    return selected


def decode_events(stdout: str | bytes | None) -> list[dict[str, Any]]:
    if isinstance(stdout, bytes):
        stdout = stdout.decode(errors="replace")
    events = []
    for line in (stdout or "").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def measure(
    args: argparse.Namespace,
    p: int,
    n: int,
    genus: int,
    parity: str,
    method: str,
    implementation: str = "current",
    model: dict[str, Any] | None = None,
) -> dict[str, Any]:
    binary = args.baseline_binary if implementation == "baseline" else args.binary
    command = [
        str(binary),
        str(p),
        str(n),
        str(genus),
        parity,
        method,
        str(args.warmups),
        str(args.repeats),
        args.goal,
    ]
    if model is not None and model.get("fixture") is not None:
        command.append(json.dumps(model["fixture"], sort_keys=True))
    started = time.perf_counter()
    timed_out = False
    progress: queue.Queue[str | None] = queue.Queue()
    lines: list[str] = []
    with tempfile.TemporaryFile(mode="w+t") as error_log:
        with subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=error_log, text=True
        ) as child:

            def read_output() -> None:
                assert child.stdout is not None
                for line in child.stdout:
                    lines.append(line)
                    progress.put(line)
                progress.put(None)

            reader = threading.Thread(target=read_output, daemon=True)
            reader.start()
            while True:
                try:
                    if progress.get(timeout=args.timeout) is None:
                        break
                except queue.Empty:
                    timed_out = True
                    child.kill()
                    break
            child.wait()
            reader.join()
            returncode = child.returncode
        error_log.seek(0)
        stderr = error_log.read()
    events = decode_events("".join(lines))
    fixture = next((event for event in events if event["event"] == "fixture"), {})
    complete = next((event for event in events if event["event"] == "complete"), {})
    samples = [
        event["seconds"] for event in events if event["event"] == "sample" and not event["warmup"]
    ]
    errors = [event["error"] for event in events if event["event"] == "error"]
    unavailable = any(
        event.get("kind") == "unavailable" for event in events if event["event"] == "error"
    )
    specification = fixture.get("model_spec", {})
    status = (
        "timeout"
        if timed_out
        else "ok"
        if complete and returncode == 0 and len(samples) == args.repeats
        else "unavailable"
        if unavailable
        else "error"
    )
    return {
        "event": "measurement",
        "model_id": model["id"] if model is not None else f"{p}:{n}:{genus}:{parity}",
        "requested_fixture": model.get("fixture") if model is not None else None,
        "family": specification.get("family", "legacy"),
        "seed": specification.get("seed", 0),
        "leading": specification.get("leading", "monic"),
        "general": specification.get("general", False),
        "prime": p,
        "residue_degree": n,
        "genus": genus,
        "parity": parity,
        "method": method,
        "implementation": implementation,
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "goal": args.goal,
        "status": status,
        "worker_wall_seconds": time.perf_counter() - started,
        "timeout_seconds": args.timeout,
        "timeout_scope": "one call without progress",
        "warmups": args.warmups,
        "samples_seconds": samples,
        "median_seconds": statistics.median(samples) if status == "ok" else None,
        "min_seconds": min(samples) if samples else None,
        "max_seconds": max(samples) if samples else None,
        "count": complete.get("count"),
        "metadata": complete.get("metadata"),
        "peak_rss_kib": complete.get("peak_rss_kib"),
        "input_sha256": hashlib.sha256(fixture.get("curve_json", "").encode()).hexdigest()
        if fixture
        else None,
        "fixture": fixture,
        "events": events,
        "returncode": returncode,
        "errors": errors,
        "stderr": stderr,
    }


def oracle(args: argparse.Namespace, record: dict[str, Any]) -> dict[str, Any]:
    """Translate exact Rust coordinates to the validation-only GP presentation."""
    curve = json.loads(record["fixture"]["curve_json"])
    field = curve["f"]["ring"]["coefficient_ring"]
    p = int(field["characteristic"])
    modulus = field["modulus"]
    if modulus is None:
        setup = f"a=Mod(1,{p});"
    else:
        coefficients = ",".join(str(int(c)) for c in modulus)
        setup = f"a=ffgen(Mod(1,{p})*Polrev([{coefficients}],z),'a);"

    def expression(polynomial: dict[str, Any]) -> str:
        terms = []
        for term in polynomial["terms"]:
            coefficient = (
                "+".join(
                    f"({int(c)})*a^{i}" for i, c in enumerate(term["coefficient"]["coefficients"])
                )
                or "0*a"
            )
            terms.append(f"({coefficient})*x^{int(term['exponent'][0])}")
        return "+".join(terms) or "0*a"

    q = int(record["fixture"]["field_order"])
    script = f"{setup}f={expression(curve['f'])};h={expression(curve['h'])};P=hyperellcharpoly(f+h^2/4);print({q}+1+polcoef(P,{2 * record['genus'] - 1}));print(Vec(P));quit\n"
    result: dict[str, Any] = {
        "event": "oracle",
        "model_id": record["model_id"],
        "prime": record["prime"],
        "residue_degree": record["residue_degree"],
        "genus": record["genus"],
        "parity": record["parity"],
        "input_sha256": record["input_sha256"],
        "script": script,
    }
    try:
        completed = subprocess.run(
            [str(args.gp), "-fq", "-s", "128M"],
            input=script,
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
        output = completed.stdout.splitlines()
        if completed.returncode or completed.stderr.strip() or len(output) != 2:
            raise ValueError(
                completed.stderr
                or completed.stdout
                or f"GP exited with status {completed.returncode}"
            )
        result.update(
            status="ok", count=str(int(output[0])), frobenius_descending=json.loads(output[1])
        )
    except (ValueError, OSError, subprocess.TimeoutExpired) as error:
        result.update(status="unverified", error=str(error))
    return result


def main() -> None:
    args = arguments()
    args.binary = args.binary.resolve(strict=True)
    if args.baseline_binary is not None:
        args.baseline_binary = args.baseline_binary.resolve(strict=True)
    if args.gp is not None:
        args.gp = args.gp.resolve(strict=True)
    models = selected_models(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "event": "provenance",
        "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest()
        if args.corpus
        else None,
        "model_specs": models,
        "revision": args.revision,
        "source_sha256": args.source_sha256,
        "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        "baseline_binary_sha256": hashlib.sha256(args.baseline_binary.read_bytes()).hexdigest()
        if args.baseline_binary
        else None,
        "baseline_source_sha256": args.baseline_source_sha256,
        "platform": platform.platform(),
        "cpu": cpu_provenance(),
        "threads": 1,
        "goal": args.goal,
        "scope": "warm public Rust full H1 Frobenius polynomial, fixture excluded"
        if args.goal == "frobenius"
        else "warm public Rust base-field point count, fixture excluded",
        "repeats": args.repeats,
        "warmups": args.warmups,
        "rustc": subprocess.run(
            ["rustc", "--version"], capture_output=True, text=True, check=True
        ).stdout.strip(),
        "oracle": str(args.gp) if args.gp else None,
        "methods": args.methods,
        "fields": args.fields,
        "genera": args.genera,
        "parities": args.parities,
        "cases": args.cases,
    }
    measurements = []
    mismatches = 0
    oracle_verified = 0
    oracle_unverified = 0
    verified_models = 0
    with args.output.open("w") as log:
        log.write(json.dumps(metadata, sort_keys=True) + "\n")
        print(json.dumps(metadata, sort_keys=True), flush=True)
        for model in models:
            p, n, genus, parity = (model[k] for k in ("prime", "residue_degree", "genus", "parity"))
            jobs = [
                (method, implementation)
                for method in model["methods"]
                for implementation in (
                    ["baseline", "current"]
                    if method == "kedlaya" and args.baseline_binary is not None
                    else ["current"]
                )
            ]
            if (genus + (parity == "even")) % 2:
                jobs.reverse()
            pair = []
            for method, implementation in jobs:
                record = measure(args, p, n, genus, parity, method, implementation, model)
                measurements.append(record)
                pair.append(record)
                log.write(json.dumps(record, sort_keys=True) + "\n")
                log.flush()
                summary = {
                    key: record[key]
                    for key in (
                        "model_id",
                        "family",
                        "seed",
                        "leading",
                        "general",
                        "prime",
                        "residue_degree",
                        "genus",
                        "parity",
                        "method",
                        "implementation",
                        "status",
                        "median_seconds",
                        "count",
                    )
                }
                print(json.dumps(summary, sort_keys=True), flush=True)
            hashes = {record["input_sha256"] for record in pair if record["fixture"]}
            returned_counts = {
                event["count"]
                for record in pair
                for event in record["events"]
                if event["event"] == "sample"
            }
            returned_polynomials = {
                tuple(event["metadata"]["frobenius_descending"])
                for record in pair
                for event in record["events"]
                if event["event"] == "sample" and "frobenius_descending" in event["metadata"]
            }
            if len(hashes) > 1 or len(returned_counts) > 1 or len(returned_polynomials) > 1:
                mismatches += 1
                log.write(
                    json.dumps(
                        {
                            "event": "mismatch",
                            "model_id": model["id"],
                            "prime": p,
                            "residue_degree": n,
                            "genus": genus,
                            "parity": parity,
                        }
                    )
                    + "\n"
                )
            if args.gp and any(record["fixture"] for record in pair):
                checked = oracle(args, next(record for record in pair if record["fixture"]))
                if checked["status"] == "ok":
                    oracle_verified += 1
                    polynomials = [
                        event["metadata"].get("frobenius_descending")
                        for record in pair
                        for event in record["events"]
                        if event["event"] == "sample"
                    ]
                    if any(count != checked["count"] for count in returned_counts) or any(
                        polynomial is not None
                        and polynomial != [str(c) for c in checked["frobenius_descending"]]
                        for polynomial in polynomials
                    ):
                        mismatches += 1
                        checked["status"] = "mismatch"
                    elif returned_polynomials or (args.goal == "point_count" and returned_counts):
                        verified_models += 1
                else:
                    oracle_unverified += 1
                log.write(json.dumps(checked, sort_keys=True) + "\n")
                log.flush()
        summary = {
            "event": "summary",
            "models": len(models),
            "verified_models": verified_models,
            "measurements": len(measurements),
            "successful": sum(r["status"] == "ok" for r in measurements),
            "mismatches": mismatches,
            "oracle_verified": oracle_verified,
            "oracle_unverified": oracle_unverified,
        }
        log.write(json.dumps(summary, sort_keys=True) + "\n")
        print(json.dumps(summary, sort_keys=True), flush=True)
    columns = [
        "model_id",
        "family",
        "seed",
        "leading",
        "general",
        "prime",
        "residue_degree",
        "genus",
        "parity",
        "method",
        "implementation",
        "binary_sha256",
        "status",
        "median_seconds",
        "min_seconds",
        "max_seconds",
        "count",
        "peak_rss_kib",
        "input_sha256",
    ]
    with args.output.with_suffix(".csv").open("w") as output:
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(measurements)
    if (
        mismatches
        or oracle_unverified
        or any(record["status"] == "error" for record in measurements)
    ):
        sys.exit(1)


if __name__ == "__main__":
    main()
