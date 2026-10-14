#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
# check-no-gpl.sh — enforce GPL-isolation (repo-root AGENTS.md invariant 1).
#
# The *conveyed* (no-gpl) engine artifact must link and reference NO GPL.
# The conveyed profile contains only native kernels and permissive/LGPL
# libraries. Optional GPL tools are subprocess adapters in noether-bridges and
# are not enabled in this artifact.
# This turns the `cargo tree` / `ldd` snapshot into an enforced gate.
#
# Builds the no-gpl private extension cdylib, then asserts on the .so:
#   1. ldd  — no PARI library among its dynamic dependencies.
#   2. nm   — no PARI symbols (pari_init / matsnf0 / qfauto0 / pari_shim_* …),
#             whether imported (undefined) or defined.
#
# The LGPL substrate (FLINT / GMP / MPFR) is expected and permitted.
#
# Usage:  scripts/check-no-gpl.sh [debug|release]   (default: debug)
# Honors FLINT_PREFIX / LD_LIBRARY_PATH from the environment.
set -euo pipefail

PROFILE="${1:-debug}"
PYPROJECT="pyproject.toml"

# Maturin's profile is authoritative for every conveyed build. The list is
# deliberately kept on one TOML line so this gate needs no Python/TOML parser
# merely to prove that Cargo and Maturin receive identical features.
FEATURES=$(sed -n \
  's/^features[[:space:]]*=[[:space:]]*\[\(.*\)\][[:space:]]*$/\1/p' \
  "$PYPROJECT" | tr -d ' "')
if [ -z "$FEATURES" ]; then
  echo "FAIL: could not read the conveyed feature list from $PYPROJECT"
  exit 1
fi
case ",$FEATURES," in
  *,pari,*|*,pari-ffi,*|*,gap-subprocess,*)
    echo "FAIL: the conveyed Maturin profile enables a forbidden GPL feature"
    exit 1
    ;;
esac

echo ">> building no-gpl engine ($PROFILE): --no-default-features --features $FEATURES"
build_args=(-p noether-python --no-default-features --features "$FEATURES")
[ "$PROFILE" = "release" ] && build_args+=(--release)
# codegen-units=1 keeps peak memory bounded (the engine is a large crate).
CARGO_PROFILE_DEV_CODEGEN_UNITS="${CARGO_PROFILE_DEV_CODEGEN_UNITS:-1}" \
CARGO_PROFILE_RELEASE_CODEGEN_UNITS="${CARGO_PROFILE_RELEASE_CODEGEN_UNITS:-1}" \
  cargo build "${build_args[@]}"

SO="target/$PROFILE/libnoether_python.so"
[ -f "$SO" ] || { echo "FAIL: built artifact $SO not found"; exit 1; }
echo ">> auditing $SO"

fail=0

# GPL shared libraries that must never be dynamically linked into a conveyed
# artifact.
gpl_libs='libpari|libumfpack|libsingular|libgap|libm4ri|libm4rie'
echo ">> [1/2] ldd — no GPL library linked"
if ldd "$SO" | grep -iE "$gpl_libs"; then
  echo "   FAIL: the conveyed artifact links a GPL library"
  fail=1
else
  echo "   ok: no GPL library among dynamic dependencies"
fi

# GPL symbols (imported 'U' or defined) — any hit means a GPL backend leaked:
#   * PARI  (pari-sys / libpari)
#   * UMFPACK  (GPL SuiteSparse sparse-LU; native spade/SuperLU is the path)
#   * CHOLMOD GPL modules  (Supernodal / Modify) — only reachable if a GPL
#     libcholmod is linked; the LGPL-only build omits them.
gpl_syms='pari_init|pari_shim|matsnf|qfauto|qfisom|qfminim|factorint|gerepile|\bavma\b'
gpl_syms="$gpl_syms|umfpack_[dz][il]_"
gpl_syms="$gpl_syms|cholmod_super_(numeric|symbolic|solve)|cholmod_(rowadd|rowdel|updown)"
gpl_syms="$gpl_syms|(^|[^[:alnum:]_])mzd_(init|mul|pluq|echelonize)([^[:alnum:]_]|$)"
gpl_syms="$gpl_syms|(^|[^[:alnum:]_])mzed_(init|mul|pluq|echelonize)([^[:alnum:]_]|$)"
echo ">> [2/2] nm — no GPL symbols (PARI / M4RI / UMFPACK / GPL-CHOLMOD modules)"
if nm -D "$SO" 2>/dev/null | grep -iE "$gpl_syms"; then
  echo "   FAIL: GPL symbols present in the conveyed artifact"
  fail=1
else
  echo "   ok: no GPL symbols"
fi

if [ "$fail" -ne 0 ]; then
  echo
  echo "GPL GATE FAILED — GPL leaked into the conveyed (no-gpl) engine."
  echo "GPL capabilities must stay behind optional noether-bridges subprocess adapters."
  echo "See repo-root AGENTS.md cross-cutting invariant 1."
  exit 1
fi
echo
echo "GPL GATE PASSED — conveyed artifact links and references no GPL (PARI / M4RI / UMFPACK / GPL-CHOLMOD)."
