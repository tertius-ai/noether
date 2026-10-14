#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
#
# Copy the non-system shared libraries the private extension links
# against into a sibling tertius_noether.libs/ directory, matching the
# $ORIGIN/../tertius_noether.libs RUNPATH that the *-sys crates bake
# into the .so at link time.
#
# After this runs, `import noether._engine` resolves libflint /
# its native libraries at load time with no LD_LIBRARY_PATH / LD_PRELOAD
# dance. Mirrors the layout that `auditwheel repair` produces for
# manylinux wheels, so the development and conveyed-wheel layouts converge.
#
# Run after `maturin develop`. Idempotent: re-running overwrites copies
# in place. Reads the .so's DT_NEEDED list rather than hard-coding the
# allowed library set. Forbidden GPL-linked libraries fail closed. An
# internal, non-conveyed build may opt in explicitly with
# `NOETHER_ALLOW_GPL_LINKED=1`; ordinary development therefore exercises the
# same license posture as the wheel.

set -euo pipefail

# macOS development builds resolve FLINT from the configured prefix. Maturin's
# wheel repair owns copying and rewriting dylibs for a conveyed wheel; this
# helper only mirrors the repaired layout for local Linux development.
if [ "$(uname -s)" = "Darwin" ]; then
    echo "[install-native-deps] macOS detected — skipping (dyld @rpath handles this)."
    exit 0
fi

PY="${PY:-.venv/bin/python}"
if [ ! -x "$PY" ]; then
    echo "[install-native-deps] ERROR: python interpreter not found at $PY"
    echo "[install-native-deps] Set PY=<path> or run from the workspace root after make sync."
    exit 1
fi

# Locate the engine package directory via Python — works for both
# editable-installed (in-tree source dir) and wheel-installed
# (site-packages) layouts. The engine is the mixed-package submodule
# `noether._engine` (maturin `module-name = "noether._engine"`,
# `python-source = "python"`), so its compiled extension lives *inside*
# the `noether` package as `_engine.cpython-3XX-<arch>.so`, next to the
# pure-Python sources. `noether._engine.__file__` points straight at that
# .so; its dirname is the package dir.
PKG_DIR=$("$PY" -c "import noether._engine as e, os; print(os.path.dirname(e.__file__))" 2>/dev/null || true)
if [ -z "$PKG_DIR" ] || [ ! -d "$PKG_DIR" ]; then
    # Fall back to the maturin develop output path inside the repo — the
    # mixed package puts the .so beside the python-source tree.
    PKG_DIR="python/noether"
fi
ENGINE_SO=$(ls "$PKG_DIR"/_engine*.so 2>/dev/null | head -1 || true)
if [ -z "$ENGINE_SO" ] || [ ! -f "$ENGINE_SO" ]; then
    echo "[install-native-deps] ERROR: noether._engine compiled extension (.so) not found."
    echo "[install-native-deps] Looked in: $PKG_DIR"
    echo "[install-native-deps] Run 'make engine' first."
    exit 1
fi
LIBS_DIR="$PKG_DIR/../tertius_noether.libs"
mkdir -p "$LIBS_DIR"

# This directory is generated entirely by this script. Synchronize it to the
# current extension rather than accumulating libraries from earlier feature
# profiles (in particular, a no-GPL rebuild must remove a stale libpari copied
# by an internal build).
find "$LIBS_DIR" -mindepth 1 -maxdepth 1 -type f -delete

# Candidate search dirs, in priority order. FLINT_PREFIX comes from the
# Makefile / install-flint.sh. System paths come last so a
# patched local build beats a stale system install.
_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEARCH_DIRS=(
    "${FLINT_PREFIX:-$_REPO_ROOT/.deps/flint}/lib"
    "${FLINT_PREFIX:-$_REPO_ROOT/.deps/flint}"
    "/usr/local/lib"
    "/usr/lib/x86_64-linux-gnu"
    "/usr/lib64"
    "/usr/lib"
)

# Read DT_NEEDED from the engine .so. Linux only; macOS wheel closure is
# handled and audited during `make wheel`.
if ! command -v readelf >/dev/null; then
    echo "[install-native-deps] ERROR: readelf not found (install binutils)."
    exit 1
fi

NEEDED=$(readelf -d "$ENGINE_SO" \
    | awk '/\(NEEDED\)/ { gsub(/[\[\]]/, "", $5); print $5 }')

forbidden=()
for lib in $NEEDED; do
    case "$lib" in
        libpari*|libumfpack*|libsingular*|libgap*|libm4ri*|libm4rie*) forbidden+=("$lib") ;;
    esac
done
if [ "${#forbidden[@]}" -gt 0 ] && [ "${NOETHER_ALLOW_GPL_LINKED:-0}" != "1" ]; then
    echo "[install-native-deps] ERROR: the engine links forbidden GPL libraries:"
    for lib in "${forbidden[@]}"; do echo "  - $lib"; done
    echo "[install-native-deps] Rebuild with the conveyed Maturin feature profile."
    echo "[install-native-deps] Internal non-conveyed builds must set NOETHER_ALLOW_GPL_LINKED=1 explicitly."
    exit 1
fi

# Libraries that always come from glibc / the loader / libgcc. Never
# bundle these — bundling glibc breaks every container and CI image.
skip_lib() {
    case "$1" in
        ld-linux*|ld-*.so*) return 0 ;;
        libc.so*|libdl.so*|libm.so*|libpthread.so*|librt.so*|libresolv.so*) return 0 ;;
        libgcc_s.so*|libstdc++.so*|libgomp.so*) return 0 ;;
        *) return 1 ;;
    esac
}

copied=0
skipped=0
missing=()
for lib in $NEEDED; do
    if skip_lib "$lib"; then
        skipped=$((skipped + 1))
        continue
    fi
    found=""
    for dir in "${SEARCH_DIRS[@]}"; do
        if [ -e "$dir/$lib" ]; then
            found="$dir/$lib"
            break
        fi
    done
    if [ -z "$found" ]; then
        missing+=("$lib")
        continue
    fi
    # cp -L resolves the symlink so we copy the real .so contents.
    # Then preserve the SONAME basename the engine links against.
    cp -L "$found" "$LIBS_DIR/$lib"
    copied=$((copied + 1))
done

echo "[install-native-deps] copied=$copied skipped=$skipped target=$LIBS_DIR"
if [ "${#missing[@]}" -gt 0 ]; then
    echo "[install-native-deps] WARNING: could not locate the following NEEDED libs:"
    for m in "${missing[@]}"; do echo "  - $m"; done
    echo "[install-native-deps] These will fall back to the system loader path"
    echo "[install-native-deps] at runtime — extend SEARCH_DIRS in this script"
    echo "[install-native-deps] (or set FLINT_PREFIX) if that"
    echo "[install-native-deps] is not what you want."
fi
