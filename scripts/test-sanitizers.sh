#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
# Exercise the FLINT ownership boundary and its facade callers under ASan.
set -euo pipefail

repo=$(git rev-parse --show-toplevel)
toolchain=${NOETHER_SANITIZER_TOOLCHAIN:-nightly-2026-06-01}
host=$(rustc "+$toolchain" -vV | sed -n 's/^host: //p')

case "$host" in
    x86_64-unknown-linux-gnu|aarch64-unknown-linux-gnu) ;;
    *)
        echo "FAIL: the AddressSanitizer gate supports Linux GNU targets, got $host"
        exit 2
        ;;
esac

if ! rustup component list --toolchain "$toolchain" \
    | grep -q '^rust-src (installed)'; then
    echo "FAIL: rust-src is required; run: rustup component add rust-src --toolchain $toolchain"
    exit 2
fi

cd "$repo"
# FLINT deliberately keeps thread-local and master caches alive until callers
# invoke flint_cleanup[_master]. A parallel libtest process cannot safely make
# that one-shot call after the harness has joined every worker, so this gate
# checks invalid accesses and ownership errors but leaves cache leak accounting
# to FLINT's own test suite.
ASAN_OPTIONS="detect_leaks=0:halt_on_error=1" \
cargo "+$toolchain" \
    --config "target.${host}.rustflags=['-Zsanitizer=address','-Cforce-frame-pointers=yes']" \
    test -Zbuild-std --target "$host" --target-dir target/asan \
    --locked -p flint-core -p noether -p noether-python --lib --tests
