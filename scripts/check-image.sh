#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
# Audit an installed Noether runtime image without trusting its build stages.
set -Eeuo pipefail

trap 'status=$?; echo "FAIL: image audit stopped at line $LINENO (status $status)" >&2' ERR

phase() {
    echo ">> auditing $1" >&2
}

usage() {
    echo "usage: scripts/check-image.sh cpu|mpi [image]" >&2
    echo "Set NOETHER_DOCKER_SUDO=1 when only Docker requires sudo access." >&2
}

profile=${1:-}
case "$profile" in
    cpu|mpi) ;;
    *) usage; exit 2 ;;
esac

repo=$(git rev-parse --show-toplevel)
version=$(sed -n 's/^version = "\([^"]*\)"/\1/p' "$repo/pyproject.toml" | head -n 1)
image=${2:-${NOETHER_IMAGE_REPOSITORY:-noether}:$version-$profile}
docker_command=(docker)
if [[ "${NOETHER_DOCKER_SUDO:-0}" == 1 ]]; then
    docker_command=(sudo docker)
fi
command -v "${docker_command[0]}" >/dev/null || {
    echo "FAIL: Docker is required to audit the reference images" >&2
    exit 2
}
for tool in nm od readelf python3; do
    command -v "$tool" >/dev/null || {
        echo "FAIL: image native audit requires host tool $tool" >&2
        exit 2
    }
done

license_hashes=$(python3 - "$repo" <<'PY'
import hashlib
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
for relative in (
    "LICENSE",
    "licenses/GPL-3.0.txt",
    "licenses/LGPL-3.0.txt",
    "licenses/NATIVE_LIBRARIES.txt",
    "licenses/README.txt",
):
    print(hashlib.sha256((root / relative).read_bytes()).hexdigest())
PY
)

phase "image identity and provenance labels"
configured_user=$("${docker_command[@]}" image inspect --format '{{.Config.User}}' "$image")
case "$configured_user" in
    ""|root|0|0:0)
        echo "FAIL: $image is configured to run as ${configured_user:-root}" >&2
        exit 1
        ;;
esac

recorded_profile=$("${docker_command[@]}" image inspect \
    --format '{{index .Config.Labels "ai.tertius.noether.profile"}}' "$image")
[[ "$recorded_profile" == "$profile" ]] || {
    echo "FAIL: $image records profile=$recorded_profile, expected $profile" >&2
    exit 1
}
recorded_features=$("${docker_command[@]}" image inspect \
    --format '{{index .Config.Labels "ai.tertius.noether.features"}}' "$image")
recorded_version=$("${docker_command[@]}" image inspect \
    --format '{{index .Config.Labels "org.opencontainers.image.version"}}' "$image")
revision=$("${docker_command[@]}" image inspect \
    --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' "$image")
dirty=$("${docker_command[@]}" image inspect \
    --format '{{index .Config.Labels "ai.tertius.noether.source-dirty"}}' "$image")
expected_revision=$(git -C "$repo" rev-parse HEAD)
[[ "$recorded_version" == "$version" ]] || {
    echo "FAIL: $image records version=$recorded_version, expected $version" >&2
    exit 1
}
[[ "$revision" == "$expected_revision" ]] || {
    echo "FAIL: $image records revision=$revision, expected current HEAD $expected_revision" >&2
    exit 1
}
[[ "$dirty" == false ]] || {
    echo "FAIL: $image was built from a dirty source tree" >&2
    exit 1
}

run_read_only() {
    "${docker_command[@]}" run --rm -i --network none --read-only \
        --tmpfs /tmp:rw,noexec,nosuid,size=64m \
        --tmpfs /work:rw,noexec,nosuid,size=64m \
        "$image" "$@"
}

phase "immutable build provenance"
run_read_only sh -eu -c '
    provenance=/usr/local/share/noether/image-build.txt
    test -f "$provenance"
    grep -Fqx "version=$1" "$provenance"
    grep -Fqx "source_revision=$2" "$provenance"
    grep -Fqx "source_dirty=false" "$provenance"
' sh "$version" "$expected_revision"

phase "read-only exact-arithmetic smoke"
run_read_only python - <<'PY'
from noether import capabilities
from noether.linalg import Matrix
from noether.numthy import FiniteField

f2 = FiniteField.prime(2)
a = Matrix.from_rows(f2, [[1, 0], [1, 1]])
assert [[int(x) for x in row] for row in (a @ a).to_rows()] == [[1, 0], [0, 1]]
assert capabilities().version
PY

phase "runtime contents and native linkage"
native_report=$(run_read_only sh -eu -c '
    for forbidden_path in \
        /build /src /crates /docs /examples /scripts /validation /work/noether
    do
        if [ -e "$forbidden_path" ]; then
            echo "FAIL: runtime image contains forbidden path $forbidden_path" >&2
            exit 1
        fi
    done
    for forbidden_command in cargo rustc cc gp gap M2 msolve Singular singular; do
        if command -v "$forbidden_command" >/dev/null; then
            echo "FAIL: runtime image contains forbidden command $forbidden_command" >&2
            exit 1
        fi
    done
    package_dir=$(python -c "import noether, os; print(os.path.dirname(noether.__file__))")
    unexpected_dir=$(find "$package_dir" -type d \
        \( -name __pycache__ -o -name .pytest_cache -o -name .hypothesis \
           -o -name tests -o -name validation -o -name docs -o -name planning \) \
        -print -quit)
    if [ -n "$unexpected_dir" ]; then
        echo "FAIL: installed package contains forbidden directory $unexpected_dir" >&2
        exit 1
    fi
    unexpected_file=$(find "$package_dir" -type f \
        ! -name "*.py" ! -name "*.pyi" ! -name "*.json" \
        ! -name "*.so" ! -name "*.pyd" ! -name py.typed \
        -print -quit)
    if [ -n "$unexpected_file" ]; then
        echo "FAIL: installed package contains unexpected file $unexpected_file" >&2
        exit 1
    fi
    for directory in "$package_dir" /opt/noether-native/lib; do
        if [ -d "$directory" ]; then
            static_payload=$(find "$directory" -type f \
                \( -iname "*.a" -o -iname "*.rlib" -o -iname "*.o" -o -iname "*.obj" \) \
                -print -quit)
            if [ -n "$static_payload" ]; then
                echo "FAIL: runtime image contains static/object payload $static_payload" >&2
                exit 1
            fi
        fi
    done
    python - "$2" "$3" "$4" "$5" "$6" <<'PY'
import hashlib
import sys
from importlib.metadata import distribution

dist = distribution("tertius-noether")
licenses = {
    str(path).split(".dist-info/licenses/", 1)[1]
    for path in dist.files or ()
    if ".dist-info/licenses/" in str(path)
}
expected = {
    "LICENSE",
    "licenses/GPL-3.0.txt",
    "licenses/LGPL-3.0.txt",
    "licenses/NATIVE_LIBRARIES.txt",
    "licenses/README.txt",
}
if licenses != expected:
    raise SystemExit(f"installed license payload is {sorted(licenses)}, expected {sorted(expected)}")
expected_hashes = dict(zip(sorted(expected), sys.argv[1:], strict=True))
for path in dist.files or ():
    text = str(path)
    marker = ".dist-info/licenses/"
    if marker not in text:
        continue
    relative = text.split(marker, 1)[1]
    digest = hashlib.sha256(dist.locate_file(path).read_bytes()).hexdigest()
    if digest != expected_hashes[relative]:
        raise SystemExit(f"installed license payload does not match repository {relative}")
PY
    share=/usr/local/share/noether
    expected="image-build.txt"
    if [ "$1" = mpi ]; then
        expected="$expected mpi-runtime.txt native-runtime.txt native-runtime.cyclonedx.json"
    fi
    actual=$(find "$share" -maxdepth 1 -type f -printf "%f\n" | sort | tr "\n" " ")
    expected=$(printf "%s\n" $expected | sort | tr "\n" " ")
    if [ "$actual" != "$expected" ]; then
        echo "FAIL: runtime share files are [$actual], expected [$expected]" >&2
        exit 1
    fi
    extension=$(python -c "from noether import _engine; print(_engine.__file__)")
    ldd "$extension"
' sh "$profile" $license_hashes)
printf '%s\n' "$native_report"
if printf '%s\n' "$native_report" \
    | grep -iE 'libpari|libumfpack|libsingular|libgap|libm4ri|libm4rie'; then
    echo "FAIL: $image links a forbidden GPL library" >&2
    exit 1
fi
if printf '%s\n' "$native_report" | grep -Fq 'not found'; then
    echo "FAIL: $image has an unresolved native dependency" >&2
    exit 1
fi

phase "native symbols, dependencies, and RPATHs"
audit_dir=$(mktemp -d)
container_id=$("${docker_command[@]}" create "$image")
cleanup_native_audit() {
    "${docker_command[@]}" rm -f "$container_id" >/dev/null 2>&1 || true
    rm -rf "$audit_dir"
}
trap cleanup_native_audit EXIT
native_paths=$(run_read_only sh -eu -c '
    extension=$(python -c "from noether import _engine; print(_engine.__file__)")
    printf "%s\n" "$extension"
    site=$(dirname "$(dirname "$extension")")
    for directory in "$site/tertius_noether.libs" /opt/noether-native/lib; do
        if [ -d "$directory" ]; then
            find "$directory" -type f -print
        fi
    done
')
native_count=0
native_names=""
while IFS= read -r native_path; do
    [[ -n "$native_path" ]] || continue
    native_count=$((native_count + 1))
    native_names="$native_names\n$(basename "$native_path")"
    candidate="$audit_dir/$native_count-$(basename "$native_path")"
    "${docker_command[@]}" cp "$container_id:$native_path" "$candidate" >/dev/null
    magic=$(od -An -tx1 -N4 "$candidate" 2>/dev/null | tr -d '[:space:]')
    [[ "$magic" == 7f454c46 ]] || {
        echo "FAIL: conveyed native path is not ELF: $native_path" >&2
        exit 1
    }
    readelf -h "$candidate" >/dev/null || {
        echo "FAIL: unreadable ELF object: $native_path" >&2
        exit 1
    }
    if basename "$native_path" \
        | grep -iE '^(lib)?(pari|umfpack|singular|gap|m4ri|m4rie)'; then
        echo "FAIL: $native_path is a forbidden GPL library" >&2
        exit 1
    fi
    if readelf -d "$candidate" 2>/dev/null \
        | grep -iE 'NEEDED.*(libpari|libumfpack|libsingular|libgap|libm4ri|libm4rie)'; then
        echo "FAIL: $native_path links a forbidden GPL library" >&2
        exit 1
    fi
    while IFS= read -r dependency; do
        case "$dependency" in
            libc.so.6|libm.so.6|libdl.so.2|libpthread.so.0|librt.so.1|libutil.so.1|\
            libgcc_s.so.1|libstdc++.so.6|ld-linux-x86-64.so.2|\
            libflint.so.*|libgmp.so.*|libmpfr.so.*|libmpi.so.*) ;;
            *) echo "FAIL: $native_path has unexpected dependency $dependency" >&2; exit 1 ;;
        esac
    done < <(readelf -d "$candidate" 2>/dev/null \
        | sed -n 's/.*NEEDED.*\[\([^]]*\)\].*/\1/p')
    if nm -D "$candidate" 2>/dev/null \
        | grep -iE 'pari_init|pari_shim|matsnf|qfauto|qfisom|qfminim|factorint|gerepile|umfpack_[dz][il]_|cholmod_super_(numeric|symbolic|solve)|cholmod_(rowadd|rowdel|updown)|(^|[^[:alnum:]_])mzd_(init|mul|pluq|echelonize)([^[:alnum:]_]|$)|(^|[^[:alnum:]_])mzed_(init|mul|pluq|echelonize)([^[:alnum:]_]|$)'; then
        echo "FAIL: $native_path contains forbidden GPL symbols" >&2
        exit 1
    fi
    while IFS= read -r rpath; do
        old_ifs=$IFS
        IFS=:
        for entry in $rpath; do
            case "$entry" in
                '$ORIGIN'|'$ORIGIN/'*|/opt/noether-native/lib) ;;
                *) echo "FAIL: $native_path contains forbidden RPATH $entry" >&2; exit 1 ;;
            esac
        done
        IFS=$old_ifs
    done < <(readelf -d "$candidate" 2>/dev/null \
        | sed -n 's/.*\(RPATH\|RUNPATH\).*\[\([^]]*\)\].*/\2/p')
done <<<"$native_paths"
if [[ "$native_count" -eq 0 ]]; then
    echo "FAIL: $image contains no audited native objects" >&2
    exit 1
fi
if [[ "$native_count" -ne 4 ]] \
    || [[ $(printf '%b\n' "$native_names" | grep -Ec '^_engine.*\.so$') -ne 1 ]] \
    || [[ $(printf '%b\n' "$native_names" | grep -Ec '^libflint.*\.so(\.[0-9]+)*$') -ne 1 ]] \
    || [[ $(printf '%b\n' "$native_names" | grep -Ec '^libgmp.*\.so(\.[0-9]+)*$') -ne 1 ]] \
    || [[ $(printf '%b\n' "$native_names" | grep -Ec '^libmpfr.*\.so(\.[0-9]+)*$') -ne 1 ]]; then
    echo "FAIL: $image native payload is not exactly engine + FLINT/GMP/MPFR" >&2
    printf '%b\n' "$native_names" >&2
    exit 1
fi

phase "$profile capability boundary"
if [[ "$profile" == cpu ]]; then
    [[ "$recorded_features" == cpu ]] || {
        echo "FAIL: $image records features=$recorded_features, expected cpu" >&2
        exit 1
    }
    run_read_only sh -eu -c '
        ! command -v mpirun >/dev/null
        ! command -v nvidia-smi >/dev/null
        ! command -v nvcc >/dev/null
        python -c "from noether import capabilities; assert \"distributed\" not in capabilities().compiled_features"
        ! python -c "from noether.linalg import DistributedMatrix, World" >/dev/null 2>&1
    '
elif [[ "$profile" == mpi ]]; then
    [[ "$recorded_features" == cpu,distributed,mpi ]] || {
        echo "FAIL: $image records features=$recorded_features, expected cpu,distributed,mpi" >&2
        exit 1
    }
    run_read_only sh -eu -c '
        command -v mpirun >/dev/null
        test -f /usr/local/share/noether/mpi-runtime.txt
        test -f /usr/local/share/noether/native-runtime.txt
        test -f /usr/local/share/noether/native-runtime.cyclonedx.json
        ! command -v nvidia-smi >/dev/null
        ! command -v nvcc >/dev/null
        for ranks in 1 2 4; do
            mpirun --oversubscribe --bind-to none -n "$ranks" sh -c '\''
                test "${OMPI_COMM_WORLD_SIZE:-}" = "$1"
                python -c "from noether import capabilities; from noether.linalg import DistributedMatrix, World; from noether.numthy import FiniteField; world = World.initialize(); matrix = DistributedMatrix.identity(world, FiniteField.prime(2), 7); assert matrix.rank() == 7; assert capabilities().mpi_runtime is not None"
            '\'' sh "$ranks"
        done
    '
    if ! printf '%s\n' "$native_report" | grep -q 'libmpi'; then
        echo "FAIL: $image does not link the Python engine to MPI" >&2
        exit 1
    fi
    "${docker_command[@]}" run --rm -i --network none --read-only \
        --mount "type=bind,src=$repo/scripts/image-native-sbom.py,dst=/tmp/image-native-sbom.py,readonly" \
        "$image" python /tmp/image-native-sbom.py check \
          /usr/local/share/noether/native-runtime.cyclonedx.json \
          --profile mpi \
          --native-dir /opt/noether-native/lib \
          --stamp /usr/local/share/noether/native-runtime.txt
fi

phase "installed image size"
size=$("${docker_command[@]}" image inspect --format '{{.Size}}' "$image")
image_id=$("${docker_command[@]}" image inspect --format '{{.Id}}' "$image")
command -v gzip >/dev/null || {
    echo "FAIL: image size recording requires gzip" >&2
    exit 2
}
compressed_size=$("${docker_command[@]}" image save "$image" | gzip -1 | wc -c)
echo "IMAGE GATE PASSED — image=$image profile=$profile uncompressed_bytes=$size compressed_archive_bytes=$compressed_size user=$configured_user"
echo "  image_id=$image_id revision=$revision source_dirty=$dirty; read-only, exact smoke, $native_count native objects audited"

case "$profile" in
    cpu) budget=$((256 * 1024 * 1024)) ;;
    mpi) budget=$((384 * 1024 * 1024)) ;;
esac
if ((size > budget)); then
    echo "WARNING: image exceeds the $profile uncompressed-size budget ($size > $budget bytes)" >&2
fi
