#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
# Audit the exact wheel that Noether would convey.
set -euo pipefail

WHEEL=${1:-}
if [ -z "$WHEEL" ] || [ ! -f "$WHEEL" ]; then
    echo "usage: scripts/check-wheel.sh path/to/tertius_noether-*.whl"
    exit 2
fi
repo=$(cd "$(dirname "$0")/.." && pwd -P)
if ! command -v unzip >/dev/null \
    || ! command -v cmp >/dev/null \
    || ! command -v nm >/dev/null \
    || ! command -v od >/dev/null \
    || ! command -v python3 >/dev/null; then
    echo "FAIL: wheel audit requires unzip, cmp, nm, od, and python3"
    exit 2
fi

name=$(basename "$WHEEL")
case "$name" in
    tertius_noether-*.whl) ;;
    *) echo "FAIL: expected one tertius-noether distribution, got $name"; exit 1 ;;
esac

case "$name" in
    *macosx*)
        command -v otool >/dev/null || { echo "FAIL: macOS wheel audit requires otool"; exit 2; }
        native_format=mach_o
        ;;
    *manylinux*|*musllinux*|*linux*)
        command -v readelf >/dev/null || { echo "FAIL: Linux wheel audit requires readelf"; exit 2; }
        native_format=elf
        ;;
    *)
        echo "FAIL: wheel audit currently supports Linux and macOS wheels, got $name"
        exit 2
        ;;
esac

entries=$(unzip -Z1 "$WHEEL")
if grep -Eq '(^/|(^|/)\.\.(/|$))' <<<"$entries"; then
    echo "FAIL: wheel contains an unsafe archive path"
    exit 1
fi

require_entry() {
    if ! grep -Eq "$1" <<<"$entries"; then
        echo "FAIL: wheel is missing $2"
        exit 1
    fi
}

require_entry '^noether/__init__\.py$' 'noether/__init__.py'
require_entry '^noether/_engine_gateway\.py$' 'the CPU preflight loader'
require_entry '^noether/_engine[^/]*\.(so|pyd)$' 'the noether._engine extension'
require_entry '^noether/API_REFERENCE_PY\.json$' 'the generated operation index'
require_entry '^noether/ENGINE_SCHEMAS\.json$' 'the generated engine schemas'
require_entry '\.dist-info/licenses?/LICENSE$' 'the MIT license'
require_entry '\.dist-info/licenses?/licenses/GPL-3\.0\.txt$' 'the GPL 3.0 text incorporated by LGPL 3.0'
require_entry '\.dist-info/licenses?/licenses/LGPL-3\.0\.txt$' 'the LGPL 3.0 text'
require_entry '\.dist-info/licenses?/licenses/NATIVE_LIBRARIES\.txt$' 'native-library source and replacement instructions'
require_entry '\.dist-info/licenses?/licenses/README\.txt$' 'the conveyed-license explanation'
require_entry '\.dist-info/sboms/native-libraries\.cyclonedx\.json$' 'the exact native-library SBOM'
require_entry '\.dist-info/sboms/noether-python\.cyclonedx\.json$' 'the Rust dependency SBOM'
python3 "$(dirname "$0")/wheel-native-sbom.py" --check "$WHEEL"

license_count=$(grep -Ec '\.dist-info/licenses?/' <<<"$entries" || true)
if [ "$license_count" -ne 5 ]; then
    echo "FAIL: wheel must contain exactly five deliberate license/notice files, found $license_count"
    exit 1
fi

dist_info_prefix=${metadata:-}
if [ -z "$dist_info_prefix" ]; then
    dist_info_prefix=$(grep -E '\.dist-info/METADATA$' <<<"$entries")
fi
dist_info_prefix=${dist_info_prefix%METADATA}
for relative in LICENSE licenses/GPL-3.0.txt licenses/LGPL-3.0.txt \
    licenses/NATIVE_LIBRARIES.txt licenses/README.txt; do
    if ! cmp -s "$repo/$relative" <(unzip -p "$WHEEL" "${dist_info_prefix}licenses/$relative"); then
        echo "FAIL: wheel license payload does not match repository $relative"
        exit 1
    fi
done

if [ "$native_format" = elf ]; then
    if ! grep -Fxq 'rustflags = ["-C", "target-cpu=x86-64-v3"]' \
        "$repo/.cargo/config.toml"; then
        echo "FAIL: the Linux wheel build no longer has the x86-64-v3 Cargo floor"
        exit 1
    fi
    loader=$(unzip -p "$WHEEL" noether/_engine_gateway.py)
    if ! grep -Fq '_require_linux_x86_64_v3()' <<<"$loader" \
        || ! grep -Fq 'Noether'"'"'s Linux x86-64 wheel requires x86-64-v3' <<<"$loader"; then
        echo "FAIL: the Linux wheel does not fail readably before loading on a pre-v3 CPU"
        exit 1
    fi
fi

require_exactly_one() {
    count=$(grep -Ec "$1" <<<"$entries" || true)
    if [ "$count" -ne 1 ]; then
        echo "FAIL: wheel must contain exactly one $2 library, found $count"
        exit 1
    fi
}

require_exactly_one '^tertius_noether\.(libs|dylibs)/libflint[^/]*\.(so(\.[0-9]+)*|dylib)$' FLINT
require_exactly_one '^tertius_noether\.(libs|dylibs)/libgmp[^/]*\.(so(\.[0-9]+)*|dylib)$' GMP
require_exactly_one '^tertius_noether\.(libs|dylibs)/libmpfr[^/]*\.(so(\.[0-9]+)*|dylib)$' MPFR

if grep -Eq '\.md$' <<<"$entries"; then
    echo "FAIL: Markdown must not ship in the wheel"
    exit 1
fi
forbidden_payload=$(grep -E \
    '(^|/)(__pycache__|\.pytest_cache|\.hypothesis|validation|tests?|docs/planning)(/|$)|\.pyc$' \
    <<<"$entries" \
    || true)
if [ -n "$forbidden_payload" ]; then
    echo "FAIL: wheel contains a test, cache, validation, or planning payload:"
    printf '  %s\n' "$forbidden_payload"
    exit 1
fi
if grep -iE '(^|/)(transgrp|transitive[-_.]?groups?|small[-_.]?groups?)(/|[._-])' \
    <<<"$entries"; then
    echo "FAIL: GAP/TransGrp-derived group catalog data must not ship in the wheel"
    exit 1
fi
if grep -iE '(^|/)(lib)?(pari|umfpack|singular|gap|m4ri|m4rie)[^/]*$' \
    <<<"$entries"; then
    echo "FAIL: wheel contains a forbidden GPL payload"
    exit 1
fi
if grep -iE '\.(a|rlib|o|obj)$' <<<"$entries"; then
    echo "FAIL: wheel contains a forbidden static archive or object"
    exit 1
fi

metadata=$(grep -E '\.dist-info/METADATA$' <<<"$entries")
if [ "$(printf '%s\n' "$metadata" | wc -l)" -ne 1 ]; then
    echo "FAIL: wheel must contain exactly one METADATA file"
    exit 1
fi
metadata_text=$(unzip -p "$WHEEL" "$metadata")
if ! grep -Fxq 'Name: tertius-noether' <<<"$metadata_text"; then
    echo "FAIL: wheel metadata is not for tertius-noether"
    exit 1
fi
if grep -qi 'tertius-noether-engine' <<<"$metadata_text"; then
    echo "FAIL: wheel still depends on the retired engine distribution"
    exit 1
fi

audit_dir=$(mktemp -d)
trap 'rm -rf "$audit_dir"' EXIT
unzip -q "$WHEEL" -d "$audit_dir"
bundled_names=$(find "$audit_dir" -type f -exec basename {} \;)

gpl_libs='libpari|libumfpack|libsingular|libgap|libm4ri|libm4rie'
gpl_syms='pari_init|pari_shim|matsnf|qfauto|qfisom|qfminim|factorint|gerepile'
gpl_syms="$gpl_syms|(^|[^[:alnum:]_])avma([^[:alnum:]_]|$)"
gpl_syms="$gpl_syms|umfpack_[dz][il]_"
gpl_syms="$gpl_syms|cholmod_super_(numeric|symbolic|solve)|cholmod_(rowadd|rowdel|updown)"
gpl_syms="$gpl_syms|(^|[^[:alnum:]_])mzd_(init|mul|pluq|echelonize)([^[:alnum:]_]|$)"
gpl_syms="$gpl_syms|(^|[^[:alnum:]_])mzed_(init|mul|pluq|echelonize)([^[:alnum:]_]|$)"

native_count=0
while IFS= read -r -d '' candidate; do
    rel=${candidate#"$audit_dir"/}
    magic=$(od -An -tx1 -N8 "$candidate" 2>/dev/null | tr -d '[:space:]')
    case "$magic" in
        213c617263683e0a|213c7468696e3e0a)
            echo "FAIL: $rel is a forbidden static archive"
            exit 1
            ;;
    esac
    if [ "$native_format" = elf ]; then
        magic=${magic:0:8}
        if [ "$magic" != 7f454c46 ]; then
            continue
        fi
        if ! readelf -h "$candidate" >/dev/null 2>&1; then
            echo "FAIL: $rel has ELF magic but no readable ELF header"
            exit 1
        fi
        native_count=$((native_count + 1))
        if readelf -d "$candidate" 2>/dev/null | grep -iE "NEEDED.*($gpl_libs)"; then
            echo "FAIL: $rel links a forbidden GPL library"
            exit 1
        fi
        if nm -D "$candidate" 2>/dev/null | grep -iE "$gpl_syms"; then
            echo "FAIL: $rel contains forbidden GPL symbols"
            exit 1
        fi
        if readelf -d "$candidate" 2>/dev/null \
            | grep -E '(RPATH|RUNPATH)' \
            | grep -Eq '(/home/|/Users/|/usr/local|/opt/)'; then
            echo "FAIL: $rel contains a build-host RPATH"
            exit 1
        fi
        while IFS= read -r rpath; do
            old_ifs=$IFS
            IFS=:
            for entry in $rpath; do
                case "$entry" in
                    '$ORIGIN'|'$ORIGIN/'*) ;;
                    *) echo "FAIL: $rel contains a non-relative RPATH entry: $entry"; exit 1 ;;
                esac
            done
            IFS=$old_ifs
        done < <(readelf -d "$candidate" 2>/dev/null \
            | sed -n 's/.*\(RPATH\|RUNPATH\).*\[\([^]]*\)\].*/\2/p')
        while IFS= read -r dependency; do
            case "$dependency" in
                libc.so.6|libm.so.6|libdl.so.2|libpthread.so.0|librt.so.1|\
                libutil.so.1|libgcc_s.so.1|ld-linux-x86-64.so.2) ;;
                *)
                    if ! grep -Fxq "$dependency" <<<"$bundled_names"; then
                        echo "FAIL: $rel depends on unbundled native library $dependency"
                        exit 1
                    fi
                    ;;
            esac
        done < <(readelf -d "$candidate" 2>/dev/null \
            | sed -n 's/.*NEEDED.*\[\([^]]*\)\].*/\1/p')
    else
        # Apple's otool exits successfully for some ordinary text/archive
        # members after printing a diagnostic, so its status is not a file-
        # type predicate. Recognize every thin/fat Mach-O byte order first;
        # then require otool to parse each object we claim to have audited.
        magic=${magic:0:8}
        case "$magic" in
            feedface|cefaedfe|feedfacf|cffaedfe|cafebabe|bebafeca|cafebabf|bfbafeca) ;;
            *) continue ;;
        esac
        if ! otool -hv "$candidate" >/dev/null 2>&1; then
            echo "FAIL: $rel has Mach-O magic but no readable Mach-O header"
            exit 1
        fi
        native_count=$((native_count + 1))
        install_id=$(otool -D "$candidate" 2>/dev/null | sed -n '2p')
        if otool -L "$candidate" 2>/dev/null | grep -iE "$gpl_libs"; then
            echo "FAIL: $rel links a forbidden GPL library"
            exit 1
        fi
        if nm -gU "$candidate" 2>/dev/null | grep -iE "$gpl_syms"; then
            echo "FAIL: $rel contains forbidden GPL symbols"
            exit 1
        fi
        while IFS= read -r rpath; do
            case "$rpath" in
                @loader_path|@loader_path/*) ;;
                *) echo "FAIL: $rel contains a non-relative LC_RPATH: $rpath"; exit 1 ;;
            esac
        done < <(otool -l "$candidate" 2>/dev/null \
            | awk '$1 == "cmd" && $2 == "LC_RPATH" { wanted=1; next } wanted && $1 == "path" { print $2; wanted=0 }')
        while IFS= read -r dependency; do
            if [ -n "$install_id" ] && [ "$dependency" = "$install_id" ]; then
                continue
            fi
            case "$dependency" in
                @loader_path/*)
                    basename=${dependency##*/}
                    if ! grep -Fxq "$basename" <<<"$bundled_names"; then
                        echo "FAIL: $rel depends on unbundled native library $dependency"
                        exit 1
                    fi
                    ;;
                /usr/lib/*|/System/Library/*) ;;
                *) echo "FAIL: $rel depends on non-system, non-bundled path $dependency"; exit 1 ;;
            esac
        done < <(otool -L "$candidate" 2>/dev/null \
            | tail -n +2 | sed 's/^[[:space:]]*//; s/[[:space:]]*(compatibility.*$//')
    fi
done < <(find "$audit_dir" -type f -print0)

if [ "$native_count" -ne 4 ]; then
    echo "FAIL: wheel native payload must be exactly engine + FLINT/GMP/MPFR; audited $native_count objects"
    exit 1
fi

file_count=$(wc -l <<<"$entries")
py_count=$(grep -Ec '^noether/.+\.py$' <<<"$entries" || true)
json_count=$(grep -Ec '^noether/.+\.json$' <<<"$entries" || true)
size=$(wc -c < "$WHEEL")

case "$name" in
    *manylinux_2_17*|*manylinux2014*|*musllinux_1_2*|*macosx*|*win*) ;;
    *manylinux*)
        echo "WARNING: $name was built for a newer manylinux baseline; build public Linux wheels in the release container"
        ;;
    *)
        echo "WARNING: $name has no portable release platform tag"
        ;;
esac

echo "WHEEL GATE PASSED — files=$file_count python=$py_count json=$json_count native=$native_count bytes=$size"
echo "  one distribution, mixed Python/Rust payload, generated indexes present, no Markdown, GPL-clean"
