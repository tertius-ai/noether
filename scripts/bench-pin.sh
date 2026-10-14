#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
# bench-pin.sh — run a benchmark pinned to a deterministic CPU.
#
# WHY THIS EXISTS
#
# Benchmarks on a machine with heterogeneous last-level caches are
# bimodal, not noisy, and the two modes are easy to mistake for a real
# win or regression. The dev box is a Ryzen 9 7950X3D: cores 0-7 (+SMT
# 16-23) carry a 96 MB 3D V-cache, cores 8-15 (+24-31) have 32 MB. A
# working set between those two sizes runs at completely different speeds
# depending on which CCD the scheduler happened to pick — measured on
# a sparse CG workload at n=250k (~32 MB working set): 27 ms on the V-cache CCD vs
# 51 ms without, i.e. nearly 2x, from the same binary. Pinned, the same
# benchmark is stable to ±0.6%.
#
# So: pin every benchmark, and record which cache it ran against.
#
# This script discovers the topology from sysfs rather than hardcoding
# it, so it is a no-op on a homogeneous machine (every CI runner) and
# still correct on a different asymmetric one.
#
# USAGE
#
#     scripts/bench-pin.sh <command> [args...]
#
#     scripts/bench-pin.sh cargo bench -p sparb
#     scripts/bench-pin.sh .venv/bin/python validation/bench/foo.py
#
# ENV
#
#   TERTIUS_BENCH_CACHE=large|small   which LLC group (default: large)
#   TERTIUS_BENCH_CPUS=core|group     pin to one core (default; most
#                                     deterministic, right for the
#                                     single-threaded benches) or to the
#                                     whole LLC group (for rayon benches)
#   TERTIUS_BENCH_CPU=<n>             pin to exactly this CPU, no discovery
#
# It prints the choice to stderr so it lands in the bench log — an
# unlabelled number from this box is not interpretable.

set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo "usage: $0 <command> [args...]" >&2
    exit 2
fi

note() { printf '[bench-pin] %s\n' "$*" >&2; }

# ── explicit override ────────────────────────────────────────────────
if [[ -n "${TERTIUS_BENCH_CPU:-}" ]]; then
    note "pinned to CPU $TERTIUS_BENCH_CPU (TERTIUS_BENCH_CPU)"
    exec taskset -c "$TERTIUS_BENCH_CPU" "$@"
fi

if ! command -v taskset >/dev/null 2>&1; then
    note "WARNING: taskset not found — running UNPINNED. On a machine with"
    note "         heterogeneous caches these numbers are not comparable."
    exec "$@"
fi

# ── discover last-level-cache groups ─────────────────────────────────
# Walk every CPU's highest cache index and collect distinct
# (shared_cpu_list, size) pairs. Uses index3 when present (typical L3),
# falling back to the largest index the CPU exposes.
declare -A GROUP_KB
for d in /sys/devices/system/cpu/cpu[0-9]*; do
    cdir=""
    for i in 4 3 2; do
        if [[ -r "$d/cache/index$i/shared_cpu_list" && -r "$d/cache/index$i/size" ]]; then
            cdir="$d/cache/index$i"
            break
        fi
    done
    [[ -n "$cdir" ]] || continue
    list=$(<"$cdir/shared_cpu_list")
    size=$(<"$cdir/size")
    kb=${size%K}
    [[ "$kb" =~ ^[0-9]+$ ]] || continue
    GROUP_KB["$list"]=$kb
done

if [[ ${#GROUP_KB[@]} -eq 0 ]]; then
    note "could not read cache topology — running UNPINNED"
    exec "$@"
fi

# ── pick the target group ────────────────────────────────────────────
want=${TERTIUS_BENCH_CACHE:-large}
best_list=""; best_kb=-1
for list in "${!GROUP_KB[@]}"; do
    kb=${GROUP_KB[$list]}
    if [[ $best_kb -lt 0 ]]; then best_list=$list; best_kb=$kb; continue; fi
    if [[ "$want" == "small" ]]; then
        (( kb < best_kb )) && { best_list=$list; best_kb=$kb; }
    else
        (( kb > best_kb )) && { best_list=$list; best_kb=$kb; }
    fi
done

# Heterogeneous iff the groups don't all report the same size.
hetero=0
for list in "${!GROUP_KB[@]}"; do
    [[ ${GROUP_KB[$list]} -ne $best_kb ]] && hetero=1
done

# ── pick a CPU inside it ─────────────────────────────────────────────
# Expand "0-7,16-23" and prefer the first *physical* core (a CPU that is
# the lowest-numbered of its SMT siblings), so we don't land on a
# hyperthread whose sibling is doing someone else's work.
expand() {
    local spec=$1 part lo hi
    IFS=',' read -ra parts <<< "$spec"
    for part in "${parts[@]}"; do
        if [[ "$part" == *-* ]]; then
            lo=${part%-*}; hi=${part#*-}
            for ((c = lo; c <= hi; c++)); do echo "$c"; done
        else
            echo "$part"
        fi
    done
}

chosen=""
while read -r c; do
    sib="/sys/devices/system/cpu/cpu$c/topology/thread_siblings_list"
    if [[ -r "$sib" ]]; then
        first=$(<"$sib"); first=${first%%,*}; first=${first%%-*}
        [[ "$c" == "$first" ]] || continue
    fi
    chosen=$c
    break
done < <(expand "$best_list")
[[ -n "$chosen" ]] || chosen=$(expand "$best_list" | head -1)

mb=$((best_kb / 1024))
if [[ $hetero -eq 1 ]]; then
    note "heterogeneous LLC detected — pinning is REQUIRED for comparable numbers"
    note "  groups:"
    for list in "${!GROUP_KB[@]}"; do
        note "    cpus $list -> $((GROUP_KB[$list] / 1024)) MB"
    done
fi

if [[ "${TERTIUS_BENCH_CPUS:-core}" == "group" ]]; then
    note "pinned to LLC group $best_list (${mb} MB)"
    target=$best_list
else
    note "pinned to CPU $chosen (LLC group $best_list, ${mb} MB)"
    target=$chosen
fi

# ── frequency policy ─────────────────────────────────────────────────
# Report only when the policy would actually depress clocks. On
# `amd-pstate-epp` in active mode the governor STRING is not the
# meaningful knob — "powersave" there is hardware-managed CPPC, not the
# old downclocking governor, and it boosts normally (measured on this box:
# 4.8-5.0 GHz under sustained load, against a 5.76 GHz single-core peak).
# What matters is the energy-performance preference. Warning on the
# governor name alone cries wolf.
cfd=/sys/devices/system/cpu/cpu"$chosen"/cpufreq
drv=$(cat "$cfd/scaling_driver" 2>/dev/null || true)
gov=$(cat "$cfd/scaling_governor" 2>/dev/null || true)
epp=$(cat "$cfd/energy_performance_preference" 2>/dev/null || true)
pstate_mode=$(cat /sys/devices/system/cpu/amd_pstate/status 2>/dev/null || true)

if [[ "$drv" == amd-pstate* && "$pstate_mode" == "active" ]]; then
    case "$epp" in
        performance | balance_performance | "")
            note "freq policy: $drv, EPP=$epp (boosts normally; governor name is not the knob here)"
            ;;
        *)
            note "NOTE: $drv EPP=$epp is a power-SAVING preference — clocks may be"
            note "      held down. Prefer EPP=performance for benchmarking (needs root)."
            ;;
    esac
elif [[ -n "$gov" && "$gov" != "performance" ]]; then
    note "NOTE: scaling_governor=$gov (not 'performance') — expect some"
    note "      frequency-scaling variance; changing it needs root."
fi

exec taskset -c "$target" "$@"
