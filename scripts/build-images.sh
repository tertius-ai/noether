#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
# Build Noether's named reference OCI images from a bounded source context.
set -euo pipefail

usage() {
    cat <<'EOF'
usage: scripts/build-images.sh [--worktree] [cpu|mpi|all]

By default the input is a clean committed HEAD. --worktree stages tracked and
unignored files as they currently exist and records source-dirty=true.
Set NOETHER_DOCKER_SUDO=1 when only Docker requires sudo access.
EOF
}

source_mode=commit
target=cpu
while (($#)); do
    case "$1" in
        --worktree) source_mode=worktree ;;
        cpu|mpi|all) target=$1 ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
    shift
done

repo=$(git rev-parse --show-toplevel)
docker_command=(docker)
if [[ "${NOETHER_DOCKER_SUDO:-0}" == 1 ]]; then
    docker_command=(sudo docker)
fi
command -v "${docker_command[0]}" >/dev/null || {
    echo "FAIL: Docker is required to build the reference images" >&2
    exit 2
}
[[ -f "$repo/docker/Dockerfile" ]] || {
    echo "FAIL: missing $repo/docker/Dockerfile" >&2
    exit 2
}

source_rev=$(git -C "$repo" rev-parse HEAD)
source_dirty=false
stage=$(mktemp -d "${TMPDIR:-/tmp}/noether-image.XXXXXX")
file_list=$(mktemp "${TMPDIR:-/tmp}/noether-image-files.XXXXXX")
trap 'rm -rf "$stage"; rm -f "$file_list"' EXIT

if [[ "$source_mode" == commit ]]; then
    if [[ -n "$(git -C "$repo" status --porcelain)" ]]; then
        echo "FAIL: committed image input requires a clean tree; use --worktree for a development build" >&2
        exit 2
    fi
    git -C "$repo" archive --format=tar HEAD | tar -xf - -C "$stage"
else
    source_dirty=true
    while IFS= read -r -d '' path; do
        if [[ -e "$repo/$path" || -L "$repo/$path" ]]; then
            printf '%s\0' "$path"
        fi
    done < <(git -C "$repo" ls-files --cached --others --exclude-standard -z) > "$file_list"
    tar -C "$repo" --null --files-from="$file_list" -cf - | tar -xf - -C "$stage"
fi

version=$(sed -n 's/^version = "\([^"]*\)"/\1/p' "$stage/pyproject.toml" | head -n 1)
[[ -n "$version" ]] || {
    echo "FAIL: could not read the package version from pyproject.toml" >&2
    exit 2
}
repository=${NOETHER_IMAGE_REPOSITORY:-noether}

build_target() {
    local profile=$1
    local tag="$repository:$version-$profile"
    if ! "${docker_command[@]}" info >/dev/null 2>&1; then
        echo "FAIL: Docker is installed, but this user cannot reach a running daemon" >&2
        exit 2
    fi
    echo ">> building $tag from $source_rev (dirty=$source_dirty)"
    "${docker_command[@]}" build \
        --file "$stage/docker/Dockerfile" \
        --target "$profile" \
        --build-arg "NOETHER_VERSION=$version" \
        --build-arg "NOETHER_SOURCE_REV=$source_rev" \
        --build-arg "NOETHER_SOURCE_DIRTY=$source_dirty" \
        --tag "$tag" \
        "$stage"
    echo "$tag"
}

case "$target" in
    cpu) build_target cpu ;;
    mpi) build_target mpi ;;
    all)
        build_target cpu
        build_target mpi
        ;;
esac
