#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
# Build a checksum-pinned GMP/MPFR/FLINT stack into one reusable prefix.
#
# Usage:
#   scripts/install-flint.sh
#   FLINT_PREFIX=/opt/flint scripts/install-flint.sh
#
# The prefix is recorded in .deps/flint/prefix.txt for Make. Direct Cargo
# callers may export FLINT_PREFIX and prepend its lib/pkgconfig directory to
# PKG_CONFIG_PATH.

set -euo pipefail

COPY_SOURCES_DIR=""
if [[ $# -gt 0 ]]; then
    if [[ $# -ne 2 || "$1" != --copy-sources || -z "$2" ]]; then
        echo "usage: scripts/install-flint.sh [--copy-sources OUTPUT_DIR]" >&2
        exit 2
    fi
    COPY_SOURCES_DIR="$2"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

GMP_VERSION="${GMP_VERSION:-6.3.0}"
GMP_SHA256="${GMP_SHA256:-a3c2b80201b89e68616f4ad30bc66aee4927c3ce50e33929ca819d5c43538898}"
GMP_ARCHIVE_URL="${GMP_ARCHIVE_URL:-https://ftp.gnu.org/gnu/gmp/gmp-${GMP_VERSION}.tar.xz}"
MPFR_VERSION="${MPFR_VERSION:-4.2.2}"
MPFR_SHA256="${MPFR_SHA256:-b67ba0383ef7e8a8563734e2e889ef5ec3c3b898a01d00fa0a6869ad81c6ce01}"
MPFR_ARCHIVE_URL="${MPFR_ARCHIVE_URL:-https://www.mpfr.org/mpfr-${MPFR_VERSION}/mpfr-${MPFR_VERSION}.tar.xz}"
FLINT_VERSION="${FLINT_VERSION:-3.6.0}"
FLINT_SHA256="${FLINT_SHA256:-b95e2c7792f5eea4a1c8d2d42c4098434756832e57a094b295eb5dfdc9b4c36b}"
FLINT_ARCHIVE_URL="${FLINT_ARCHIVE_URL:-https://github.com/flintlib/flint/releases/download/v${FLINT_VERSION}/flint-${FLINT_VERSION}.tar.gz}"

FLINT_PREFIX="${FLINT_PREFIX:-$REPO_ROOT/.deps/flint}"
FLINT_CFLAGS="${FLINT_CFLAGS:-${CFLAGS:--O3}}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-$REPO_ROOT/.deps/downloads}"
SOURCE_DIR="${SOURCE_DIR:-$REPO_ROOT/.deps/native-src}"
PREFIX_NOTE_FILE="${PREFIX_NOTE_FILE:-$REPO_ROOT/.deps/flint/prefix.txt}"
STAMP_FILE="$FLINT_PREFIX/.tertius-native-build"
RUN_NATIVE_TESTS="${RUN_NATIVE_TESTS:-${RUN_FLINT_TESTS:-1}}"
JOBS="${JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"

if [[ "$(uname -s)" == Darwin ]] && [[ -z "${MACOSX_DEPLOYMENT_TARGET:-}" ]]; then
    case "$(uname -m)" in
        arm64) MACOSX_DEPLOYMENT_TARGET=11.0 ;;
        *) MACOSX_DEPLOYMENT_TARGET=10.13 ;;
    esac
    export MACOSX_DEPLOYMENT_TARGET
fi

log() { printf '[install-flint] %s\n' "$*" >&2; }
die() { printf '[install-flint] ERROR: %s\n' "$*" >&2; exit 1; }

compiler_identity() {
    local compiler="${CC:-cc}"
    "$compiler" --version 2>/dev/null | sed -n '1p'
}

stamp_contents() {
    printf '%s\n' \
        'version=7' \
        "gmp_version=$GMP_VERSION" \
        "gmp_source_sha256=$GMP_SHA256" \
        "mpfr_version=$MPFR_VERSION" \
        "mpfr_source_sha256=$MPFR_SHA256" \
        "flint_version=$FLINT_VERSION" \
        "flint_source_sha256=$FLINT_SHA256" \
        "prefix=$FLINT_PREFIX" \
        "system=$(uname -s)" \
        "machine=$(uname -m)" \
        "macosx_deployment_target=${MACOSX_DEPLOYMENT_TARGET:-}" \
        "compiler=$(compiler_identity)" \
        "cppflags=${CPPFLAGS:-}" \
        "cflags=${CFLAGS:-}" \
        "flint_cflags=$FLINT_CFLAGS" \
        "ldflags=${LDFLAGS:-}" \
        "native_tests=$RUN_NATIVE_TESTS"
}

shared_library_exists() {
    [[ -f "$FLINT_PREFIX/lib/lib$1.so" || -f "$FLINT_PREFIX/lib/lib$1.dylib" ]]
}

complete_install_exists() {
    shared_library_exists gmp \
        && shared_library_exists mpfr \
        && shared_library_exists flint
}

checksum() {
    if command -v sha256sum >/dev/null; then
        sha256sum "$1" | awk '{print $1}'
    elif command -v shasum >/dev/null; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        die "sha256sum or shasum is required"
    fi
}

download_archive() {
    local name="$1"
    local url="$2"
    local expected="$3"
    local archive="$4"
    mkdir -p "$DOWNLOAD_DIR"
    local temporary="$archive.download.$$"
    trap 'rm -f "$temporary"' RETURN
    log "downloading $name release archive"
    if command -v curl >/dev/null; then
        local curl_args=(
            --fail --location --silent --show-error
            --connect-timeout 20 --retry 4
        )
        # curl 7.71 introduced --retry-all-errors, but manylinux2014's
        # deliberately old userspace predates it.  Preserve the stronger retry
        # policy where available without raising the wheel's compatibility
        # floor merely to download checksum-pinned sources.
        if curl --help all 2>/dev/null | grep -q -- '--retry-all-errors'; then
            curl_args+=(--retry-all-errors)
        fi
        curl "${curl_args[@]}" --output "$temporary" "$url"
    elif command -v wget >/dev/null; then
        wget --quiet --output-document="$temporary" "$url"
    else
        die "curl or wget is required"
    fi
    local actual
    actual="$(checksum "$temporary")"
    [[ "$actual" == "$expected" ]] \
        || die "$name archive checksum mismatch: expected $expected, got $actual"
    mv -f "$temporary" "$archive"
    trap - RETURN
}

verified_archive() {
    local name="$1"
    local url="$2"
    local expected="$3"
    local filename="$4"
    local archive="$DOWNLOAD_DIR/$filename"
    if [[ ! -f "$archive" ]] || [[ "$(checksum "$archive")" != "$expected" ]]; then
        if [[ -f "$archive" ]]; then
            log "discarding cached $name archive with the wrong checksum"
        fi
        download_archive "$name" "$url" "$expected" "$archive"
    fi
    log "verified $name source sha256=$expected"
    printf '%s\n' "$archive"
}

prepare_source() {
    local name="$1"
    local version="$2"
    local expected="$3"
    local archive="$4"
    local source="$SOURCE_DIR/$name-$version-${expected:0:16}"
    local source_stamp="$source/.tertius-native-source"
    local stamp="version=$version sha256=$expected"
    if [[ -f "$source_stamp" ]] && [[ "$(cat "$source_stamp")" == "$stamp" ]]; then
        printf '%s\n' "$source"
        return
    fi

    mkdir -p "$SOURCE_DIR"
    local staging
    staging="$(mktemp -d "$SOURCE_DIR/.extract.XXXXXX")"
    trap 'rm -rf "$staging"' RETURN
    tar -xf "$archive" -C "$staging"
    local extracted="$staging/$name-$version"
    [[ -x "$extracted/configure" ]] \
        || die "$name release archive does not contain $name-$version/configure"
    printf '%s\n' "$stamp" > "$extracted/.tertius-native-source"
    if [[ -e "$source" ]]; then
        local backup="$source.stale.$(date +%Y%m%d%H%M%S).$$"
        log "moving nonmatching $name source tree to $backup"
        mv "$source" "$backup"
    fi
    mv "$extracted" "$source"
    rmdir "$staging"
    trap - RETURN
    printf '%s\n' "$source"
}

reset_source() {
    if [[ -f "$1/Makefile" ]]; then
        log "resetting the checksum-keyed $(basename "$1") source tree"
        make -C "$1" distclean >/dev/null 2>&1 || true
    fi
}

run_native_check() {
    local name="$1"
    case "$RUN_NATIVE_TESTS" in
        1)
            log "running the upstream $name test suite"
            make -j"$JOBS" check
            ;;
        0)
            log "skipping the upstream $name suite because RUN_NATIVE_TESTS=0"
            ;;
        *) die "RUN_NATIVE_TESTS must be 0 or 1" ;;
    esac
}

record_completed_install() {
    local stamp_tmp="$STAMP_FILE.tmp.$$"
    mkdir -p "$FLINT_PREFIX" "$(dirname "$PREFIX_NOTE_FILE")"
    printf '%s\n' "$FLINT_PREFIX" > "$PREFIX_NOTE_FILE"
    stamp_contents > "$stamp_tmp"
    mv "$stamp_tmp" "$STAMP_FILE"
}

if [[ -n "$COPY_SOURCES_DIR" ]]; then
    command -v tar >/dev/null || die "tar is required"
    gmp_archive="$(verified_archive GMP "$GMP_ARCHIVE_URL" "$GMP_SHA256" "gmp-$GMP_VERSION.tar.xz")"
    mpfr_archive="$(verified_archive MPFR "$MPFR_ARCHIVE_URL" "$MPFR_SHA256" "mpfr-$MPFR_VERSION.tar.xz")"
    flint_archive="$(verified_archive FLINT "$FLINT_ARCHIVE_URL" "$FLINT_SHA256" "flint-$FLINT_VERSION.tar.gz")"
    mkdir -p "$COPY_SOURCES_DIR"
    cp -f "$gmp_archive" "$mpfr_archive" "$flint_archive" "$COPY_SOURCES_DIR/"
    cat >"$COPY_SOURCES_DIR/SOURCE_URLS.txt" <<EOF
gmp-$GMP_VERSION.tar.xz $GMP_ARCHIVE_URL
mpfr-$MPFR_VERSION.tar.xz $MPFR_ARCHIVE_URL
flint-$FLINT_VERSION.tar.gz $FLINT_ARCHIVE_URL
EOF
    cat >"$COPY_SOURCES_DIR/SHA256SUMS" <<EOF
$GMP_SHA256  gmp-$GMP_VERSION.tar.xz
$MPFR_SHA256  mpfr-$MPFR_VERSION.tar.xz
$FLINT_SHA256  flint-$FLINT_VERSION.tar.gz
EOF
    unexpected="$(find "$COPY_SOURCES_DIR" -mindepth 1 -maxdepth 1 \
        ! -name "gmp-$GMP_VERSION.tar.xz" \
        ! -name "mpfr-$MPFR_VERSION.tar.xz" \
        ! -name "flint-$FLINT_VERSION.tar.gz" \
        ! -name SOURCE_URLS.txt ! -name SHA256SUMS -print -quit)"
    [[ -z "$unexpected" ]] || die "unexpected file in source conveyance directory: $unexpected"
    log "copied checksum-verified native sources to $COPY_SOURCES_DIR"
    exit 0
fi

if complete_install_exists \
    && [[ -f "$STAMP_FILE" ]] \
    && [[ "$(cat "$STAMP_FILE")" == "$(stamp_contents)" ]]; then
    printf '%s\n' "$FLINT_PREFIX" > "$PREFIX_NOTE_FILE"
    log "GMP $GMP_VERSION, MPFR $MPFR_VERSION, and FLINT $FLINT_VERSION are already verified at $FLINT_PREFIX"
    exit 0
fi

command -v make >/dev/null || die "make is required"
command -v tar >/dev/null || die "tar is required"
command -v pkg-config >/dev/null || die "pkg-config is required"
command -v "${CC:-cc}" >/dev/null || die "C compiler ${CC:-cc} was not found"

gmp_archive="$(verified_archive GMP "$GMP_ARCHIVE_URL" "$GMP_SHA256" "gmp-$GMP_VERSION.tar.xz")"
mpfr_archive="$(verified_archive MPFR "$MPFR_ARCHIVE_URL" "$MPFR_SHA256" "mpfr-$MPFR_VERSION.tar.xz")"
flint_archive="$(verified_archive FLINT "$FLINT_ARCHIVE_URL" "$FLINT_SHA256" "flint-$FLINT_VERSION.tar.gz")"
gmp_source="$(prepare_source gmp "$GMP_VERSION" "$GMP_SHA256" "$gmp_archive")"
mpfr_source="$(prepare_source mpfr "$MPFR_VERSION" "$MPFR_SHA256" "$mpfr_archive")"
flint_source="$(prepare_source flint "$FLINT_VERSION" "$FLINT_SHA256" "$flint_archive")"

mkdir -p "$FLINT_PREFIX"
native_cppflags="-I$FLINT_PREFIX/include${CPPFLAGS:+ $CPPFLAGS}"
native_ldflags="-L$FLINT_PREFIX/lib${LDFLAGS:+ $LDFLAGS}"
export LD_LIBRARY_PATH="$FLINT_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export DYLD_LIBRARY_PATH="$FLINT_PREFIX/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"

reset_source "$gmp_source"
gmp_config=(--prefix="$FLINT_PREFIX" --disable-static --enable-shared)
case "$(uname -m)" in
    x86_64|amd64) gmp_config+=(--enable-fat) ;;
esac
log "configuring shared GMP $GMP_VERSION at $FLINT_PREFIX"
(cd "$gmp_source" && ./configure "${gmp_config[@]}")
log "building GMP with $JOBS jobs"
make -C "$gmp_source" -j"$JOBS"
(cd "$gmp_source" && run_native_check GMP)
make -C "$gmp_source" install

reset_source "$mpfr_source"
log "configuring shared MPFR $MPFR_VERSION at $FLINT_PREFIX"
(cd "$mpfr_source" && \
    CPPFLAGS="$native_cppflags" LDFLAGS="$native_ldflags" \
    ./configure --prefix="$FLINT_PREFIX" --disable-static --enable-shared \
        --enable-thread-safe \
        --with-gmp="$FLINT_PREFIX")
log "building MPFR with $JOBS jobs"
make -C "$mpfr_source" -j"$JOBS"
(cd "$mpfr_source" && run_native_check MPFR)
make -C "$mpfr_source" install

reset_source "$flint_source"
log "configuring shared FLINT $FLINT_VERSION at $FLINT_PREFIX"
(cd "$flint_source" && \
    CPPFLAGS="$native_cppflags" CFLAGS="$FLINT_CFLAGS" LDFLAGS="$native_ldflags" \
    ./configure --prefix="$FLINT_PREFIX" --disable-static --enable-shared --disable-debug \
        --with-gmp="$FLINT_PREFIX" --with-mpfr="$FLINT_PREFIX")
log "building FLINT with $JOBS jobs"
make -C "$flint_source" -j"$JOBS"
(cd "$flint_source" && \
    LD_LIBRARY_PATH="$flint_source${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    DYLD_LIBRARY_PATH="$flint_source${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" \
    run_native_check FLINT)
make -C "$flint_source" install

complete_install_exists \
    || die "build succeeded but the shared GMP/MPFR/FLINT stack is incomplete under $FLINT_PREFIX/lib"

native_pkg_config="$FLINT_PREFIX/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
PKG_CONFIG_PATH="$native_pkg_config" pkg-config --exact-version="$GMP_VERSION" gmp \
    || die "installed pkg-config metadata does not report GMP $GMP_VERSION"
PKG_CONFIG_PATH="$native_pkg_config" pkg-config --exact-version="$MPFR_VERSION" mpfr \
    || die "installed pkg-config metadata does not report MPFR $MPFR_VERSION"
mpfr_probe="$(mktemp "${TMPDIR:-/tmp}/noether-mpfr-tls.XXXXXX")"
trap 'rm -f "$mpfr_probe"' EXIT
printf '%s\n' \
    '#include <mpfr.h>' \
    'int main(void) { return mpfr_buildopt_tls_p() == 1 ? 0 : 1; }' \
    | "${CC:-cc}" -x c -o "$mpfr_probe" - \
        $(PKG_CONFIG_PATH="$native_pkg_config" pkg-config --cflags --libs mpfr)
"$mpfr_probe" || die "installed MPFR was not built with thread-local storage"
rm -f "$mpfr_probe"
trap - EXIT
PKG_CONFIG_PATH="$native_pkg_config" pkg-config --exact-version="$FLINT_VERSION" flint \
    || die "installed pkg-config metadata does not report FLINT $FLINT_VERSION"

record_completed_install
log "verified the native stack and recorded $PREFIX_NOTE_FILE"

cat <<EOF

Direct Cargo usage:
  export FLINT_PREFIX=$FLINT_PREFIX
  export PKG_CONFIG_PATH=$FLINT_PREFIX/lib/pkgconfig\${PKG_CONFIG_PATH:+:\$PKG_CONFIG_PATH}

EOF
