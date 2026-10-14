# syntax=docker/dockerfile:1.7
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam

# Build the one repaired abi3 wheel at the manylinux floor. This image and the
# release-wheel script deliberately share the pinned manylinux base.
FROM quay.io/pypa/manylinux2014_x86_64@sha256:95440e0e72dd3a81dc8d2cf59a84d57af661456620f5bc821ff92048d0e54ff9 AS wheel-builder

ARG RUST_VERSION=1.94.0
ARG BUILD_JOBS=8
ENV CARGO_BUILD_JOBS=${BUILD_JOBS} \
    CARGO_HOME=/root/.cargo \
    RUSTUP_HOME=/root/.rustup \
    PATH=/root/.cargo/bin:/tmp/noether-bootstrap/bin:${PATH}

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --profile minimal --default-toolchain "${RUST_VERSION}" \
 && rustc --version --verbose

RUN /opt/python/cp310-cp310/bin/python -m venv /tmp/noether-bootstrap \
 && /tmp/noether-bootstrap/bin/python -m pip install \
      --disable-pip-version-check --no-cache-dir 'uv==0.11.16'

WORKDIR /work/noether

# Native arithmetic changes much less often than Noether. Keep this layer
# ahead of the source copy so ordinary Rust/Python edits reuse it.
COPY scripts/install-flint.sh scripts/install-flint.sh
RUN FLINT_PREFIX=/opt/noether-native \
    PREFIX_NOTE_FILE=/work/noether/.deps/flint/prefix.txt \
    JOBS="${BUILD_JOBS}" \
    RUN_NATIVE_TESTS=0 \
    scripts/install-flint.sh

ENV FLINT_PREFIX=/opt/noether-native \
    PKG_CONFIG_PATH=/opt/noether-native/lib/pkgconfig \
    LD_LIBRARY_PATH=/opt/noether-native/lib

COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv venv --python /opt/python/cp310-cp310/bin/python .venv \
 && VIRTUAL_ENV=/work/noether/.venv uv sync \
      --active --python /opt/python/cp310-cp310/bin/python \
      --frozen --inexact --no-install-project --group dev

COPY .cargo .cargo
COPY Cargo.toml Cargo.lock rust-toolchain.toml Makefile ./
COPY crates crates
COPY examples examples
COPY licenses licenses
COPY python python
COPY scripts scripts
COPY validation validation

RUN make wheel WHEEL_DIR=/wheelhouse \
      MATURIN_WHEEL_ARGS='--compatibility manylinux2014' \
 && wheel="$(find /wheelhouse -maxdepth 1 -type f \
      -name 'tertius_noether-*.whl' -print)" \
 && test "$(printf '%s\n' "${wheel}" | grep -c .)" -eq 1 \
 && sha256sum "${wheel}" > /wheelhouse/SHA256SUMS

# MPI wheels are site/runtime artifacts rather than PyPI-portable wheels. Build
# against the same Debian/Open MPI ABI installed in the reference runtime and
# leave MPI outside the wheel so mpirun and libmpi cannot drift apart.
FROM python:3.12-slim-bookworm@sha256:8a7e7cc04fd3e2bd787f7f24e22d5d119aa590d429b50c95dfe12b3abe52f48b AS mpi-wheel-builder

ARG RUST_VERSION=1.94.0
ARG BUILD_JOBS=8
ARG OPENMPI_DEBIAN_VERSION=4.1.4-3+b1
ENV CARGO_BUILD_JOBS=${BUILD_JOBS} \
    CARGO_HOME=/root/.cargo \
    RUSTUP_HOME=/root/.rustup \
    PATH=/root/.cargo/bin:/tmp/noether-bootstrap/bin:${PATH}

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      binutils \
      build-essential \
      ca-certificates \
      curl \
      libclang-dev \
      "libopenmpi-dev=${OPENMPI_DEBIAN_VERSION}" \
      make \
      pkg-config \
 && rm -rf /var/lib/apt/lists/* \
 && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --profile minimal --default-toolchain "${RUST_VERSION}" \
 && rustc --version --verbose \
 && python -m venv /tmp/noether-bootstrap \
 && /tmp/noether-bootstrap/bin/python -m pip install \
      --disable-pip-version-check --no-cache-dir 'uv==0.11.16'

WORKDIR /work/noether

COPY scripts/install-flint.sh scripts/install-flint.sh
RUN FLINT_PREFIX=/opt/noether-native \
    PREFIX_NOTE_FILE=/work/noether/.deps/flint/prefix.txt \
    JOBS="${BUILD_JOBS}" \
    RUN_NATIVE_TESTS=0 \
    scripts/install-flint.sh

ENV FLINT_PREFIX=/opt/noether-native \
    PKG_CONFIG_PATH=/opt/noether-native/lib/pkgconfig \
    LD_LIBRARY_PATH=/opt/noether-native/lib

COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv venv --python python .venv \
 && VIRTUAL_ENV=/work/noether/.venv uv sync \
      --active --python python --frozen --inexact --no-install-project --group dev

COPY .cargo .cargo
COPY Cargo.toml Cargo.lock rust-toolchain.toml Makefile ./
COPY crates crates
COPY examples examples
COPY licenses licenses
COPY python python
COPY scripts scripts
COPY validation validation

RUN make engine-mpi-portable \
 && find python/noether -type d -name __pycache__ -prune -exec rm -rf -- {} + \
 && find python/noether -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete \
 && rm -rf python/tertius_noether.libs \
 && BINDGEN_EXTRA_CLANG_ARGS="-isystem $(gcc -print-file-name=include)" \
      .venv/bin/maturin build --release --strip \
      --auditwheel skip \
      --features extension-module,abi3-py310,distributed \
      --target-dir /tmp/noether-mpi-target \
      --interpreter .venv/bin/python --out /mpi-wheelhouse \
 && wheel="$(find /mpi-wheelhouse -maxdepth 1 -type f \
      -name 'tertius_noether-*.whl' -print)" \
 && test "$(printf '%s\n' "${wheel}" | grep -c .)" -eq 1 \
 && sha256sum "${wheel}" > /mpi-wheelhouse/SHA256SUMS \
 && install -d /mpi-native-runtime/lib \
 && for library in flint gmp mpfr; do \
      soname="$(readelf -d "/opt/noether-native/lib/lib${library}.so" \
        | sed -n 's/.*SONAME.*\[\([^]]*\)\].*/\1/p')"; \
      test -n "${soname}"; \
      install -m 0755 \
        "$(readlink -f "/opt/noether-native/lib/lib${library}.so")" \
        "/mpi-native-runtime/lib/${soname}"; \
    done \
 && cp /opt/noether-native/.tertius-native-build \
      /mpi-native-runtime/native-runtime.txt

# Runtime images contain installed artifacts, never either build environment.
FROM python:3.12-slim-bookworm@sha256:8a7e7cc04fd3e2bd787f7f24e22d5d119aa590d429b50c95dfe12b3abe52f48b AS runtime-base

ARG NOETHER_VERSION=0.3.2
ARG NOETHER_SOURCE_REV=unknown
ARG NOETHER_SOURCE_DIRTY=false
LABEL org.opencontainers.image.title="Noether" \
      org.opencontainers.image.description="Deterministic scientific mathematics backed by Rust" \
      org.opencontainers.image.source="https://github.com/tertius-ai/noether" \
      org.opencontainers.image.version="${NOETHER_VERSION}" \
      org.opencontainers.image.revision="${NOETHER_SOURCE_REV}" \
      ai.tertius.noether.source-dirty="${NOETHER_SOURCE_DIRTY}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/noether

RUN groupadd --gid 10001 noether \
 && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin noether \
 && install -d -o noether -g noether /work /usr/local/share/noether \
 && { \
      printf 'version=%s\n' "${NOETHER_VERSION}"; \
      printf 'source_revision=%s\n' "${NOETHER_SOURCE_REV}"; \
      printf 'source_dirty=%s\n' "${NOETHER_SOURCE_DIRTY}"; \
    } > /usr/local/share/noether/image-build.txt \
 && chmod 0444 /usr/local/share/noether/image-build.txt

WORKDIR /work

FROM runtime-base AS cpu
LABEL ai.tertius.noether.profile="cpu" \
      ai.tertius.noether.features="cpu"

COPY --from=wheel-builder /wheelhouse/ /tmp/wheelhouse/
RUN python -m pip install --disable-pip-version-check --no-cache-dir --no-compile \
      /tmp/wheelhouse/tertius_noether-*.whl \
 && rm -rf /tmp/wheelhouse \
 && python -c 'from noether.linalg import Matrix; from noether.numthy import FiniteField; f2 = FiniteField.prime(2); a = Matrix.from_rows(f2, [[1, 0], [1, 1]]); assert [[int(x) for x in row] for row in (a @ a).to_rows()] == [[1, 0], [0, 1]]'

USER noether:noether
CMD ["python"]

# The reference MPI artifact deliberately uses one matched build/runtime ABI.
# Cluster-native builds remain the normal route when a site supplies another
# MPI, PMIx, fabric, or container integration.
FROM runtime-base AS mpi

ARG OPENMPI_DEBIAN_VERSION=4.1.4-3+b1
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      "openmpi-bin=${OPENMPI_DEBIAN_VERSION}" \
      "libopenmpi3=${OPENMPI_DEBIAN_VERSION}" \
 && rm -rf /var/lib/apt/lists/*

COPY --from=mpi-wheel-builder /mpi-native-runtime/lib/ /opt/noether-native/lib/
COPY --from=mpi-wheel-builder /mpi-native-runtime/native-runtime.txt /usr/local/share/noether/native-runtime.txt
COPY --from=mpi-wheel-builder /mpi-wheelhouse/ /tmp/wheelhouse/
COPY scripts/image-native-sbom.py /tmp/image-native-sbom.py

ENV LD_LIBRARY_PATH=/opt/noether-native/lib

RUN python -m pip install --disable-pip-version-check --no-cache-dir --no-compile \
      /tmp/wheelhouse/tertius_noether-*.whl \
 && rm -rf /tmp/wheelhouse \
 && { \
      printf 'profile=mpi\n'; \
      mpirun --version | sed -n '1,2p'; \
      dpkg-query -W -f='${Package}=${Version}\n' openmpi-bin libopenmpi3; \
    } > /usr/local/share/noether/mpi-runtime.txt \
 && python -c 'from noether import capabilities; assert "distributed" in capabilities().compiled_features; from noether.linalg import World'

RUN python /tmp/image-native-sbom.py write \
      /usr/local/share/noether/native-runtime.cyclonedx.json \
      --profile mpi \
      --native-dir /opt/noether-native/lib \
      --stamp /usr/local/share/noether/native-runtime.txt \
 && rm /tmp/image-native-sbom.py

LABEL ai.tertius.noether.profile="mpi" \
      ai.tertius.noether.features="cpu,distributed,mpi"
USER noether:noether
RUN OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    mpirun --oversubscribe -n 2 sh -c \
      'test "${OMPI_COMM_WORLD_SIZE:-}" = 2 && python -B -c "from noether.linalg import DistributedMatrix, World; from noether.numthy import FiniteField; world = World.initialize(); matrix = DistributedMatrix.identity(world, FiniteField.prime(2), 7); assert matrix.rank() == 7"'
CMD ["python"]
