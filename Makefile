FLINT_PREFIX_FILE := $(CURDIR)/.deps/flint/prefix.txt
FLINT_PREFIX ?= $(shell cat $(FLINT_PREFIX_FILE) 2>/dev/null || echo $(CURDIR)/.deps/flint)
PKG_CONFIG_PATH := $(FLINT_PREFIX)/lib/pkgconfig:$(PKG_CONFIG_PATH)

LD_LIBRARY_PATH := $(FLINT_PREFIX)/lib:$(FLINT_PREFIX):$(LD_LIBRARY_PATH)
DYLD_LIBRARY_PATH := $(FLINT_PREFIX)/lib:$(FLINT_PREFIX):$(DYLD_LIBRARY_PATH)
PATH := $(HOME)/.local/bin:$(PATH)
PYTHONPATH := $(CURDIR)/python:$(PYTHONPATH)
export FLINT_PREFIX PKG_CONFIG_PATH LD_LIBRARY_PATH DYLD_LIBRARY_PATH PATH PYTHONPATH

VENV := .venv
PY := $(VENV)/bin/python
PYTHON_LINK_DIR := $(shell $(PY) -c "import sysconfig; print(sysconfig.get_config_var('LIBPL') or sysconfig.get_config_var('LIBDIR') or '')" 2>/dev/null)
CARGO_TEST_ENV := LIBRARY_PATH=$(PYTHON_LINK_DIR):$${LIBRARY_PATH:-}
# Test and coverage binaries need release-mode semantics, but artifact-grade
# ThinLTO only lengthens the local feedback loop. Wheels, images, and the GPL
# linkage gate continue to use Cargo.toml's exact release profile.
CARGO_RELEASE_TEST_ENV := $(CARGO_TEST_ENV) \
	CARGO_PROFILE_RELEASE_LTO=false \
	CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16

ifeq ($(shell uname -s),Darwin)
FLINT_SO := $(firstword $(wildcard $(FLINT_PREFIX)/lib/libflint*.dylib $(FLINT_PREFIX)/libflint*.dylib))
TEST_PRELOAD :=
else
FLINT_SO := $(firstword $(wildcard $(FLINT_PREFIX)/lib/libflint.so.* $(FLINT_PREFIX)/libflint.so.*))
TEST_PRELOAD := LD_PRELOAD=$(FLINT_SO)
endif

WHEEL_DIR ?= $(CURDIR)/target/wheels
WHEEL_OUTPUT := $(abspath $(WHEEL_DIR))
MATURIN_WHEEL_ARGS ?=
RUST_HOST := $(shell rustc -vV 2>/dev/null | sed -n 's/^host: //p')
NATIVE_CARGO_CONFIG := target.$(RUST_HOST).rustflags=["-C","target-cpu=native"]
# Some distro clang executables are present without their resource headers.
# GCC's builtin include directory is a valid system fallback for bindgen in
# both that case and a complete clang installation.
MPI_BINDGEN_FALLBACK := $(shell if command -v gcc >/dev/null 2>&1; then printf '%s' "-isystem $$(gcc -print-file-name=include)"; fi)
BINDGEN_EXTRA_CLANG_ARGS ?= $(MPI_BINDGEN_FALLBACK)
export BINDGEN_EXTRA_CLANG_ARGS

.PHONY: help
help:
	@echo "Noether development commands"
	@echo
	@echo "  make sync          install Python development tools (engine separately)"
	@echo "  make engine        build the engine at the wheel CPU baseline"
	@echo "  make engine-native build the engine with target-cpu=native"
	@echo "  make engine-mpi    build the baseline engine with system MPI"
	@echo "  make engine-mpi-native build the MPI engine with target-cpu=native"
	@echo "  make docs          build the user guide, examples, and Python reference"
	@echo "  make docs-rust     build the public Rust facade reference locally"
	@echo "  make docs-mpi      build the local Rust reference with system MPI"
	@echo "  make wheel         build and audit one tertius-noether wheel"
	@echo "  make candidate-linux build and fully verify the manylinux wheel"
	@echo "  make candidate-macos build and fully verify one native macOS wheel"
	@echo "  make sdist         build and audit the source-install archive"
	@echo "  make image-cpu     build the minimal CPU reference image"
	@echo "  make image-mpi     build the MPI-enabled reference image"
	@echo "  make check-image-cpu audit the installed CPU image"
	@echo "  make check-image-mpi audit the MPI-enabled reference image"
	@echo "  make check-wheel-pythons clean-install an abi3 wheel on CPython 3.10-3.14"
	@echo "  make check-version verify one release version across all artifacts"
	@echo "  make test          run the smoke tests"
	@echo "  make test-fast     run Python tests except slow/benchmark cases"
	@echo "  make test-examples execute the 40 pure-Noether mathematics examples"
	@echo "  make test-corpus   execute the curated pure-Noether acceptance corpus"
	@echo "  make test-mpi      run the 1/2/4-rank distributed matrix gates"
	@echo "  make coverage-py   measure Python line and branch coverage"
	@echo "  make coverage-rust measure Rust workspace coverage"
	@echo "  make test-all      run the local pre-PR gate"
	@echo "  make test-stress   run ignored, slow, and expanded property tests"
	@echo "  make validate      run enabled external-CAS parity tests"
	@echo "  make validate-sage require Sage and run every Sage parity lane"
	@echo "  make validate-magma-release build native and run the strict Magma release gate"
	@echo "  make audit         audit the locked Rust dependency graph"
	@echo "  make test-msrv     compile all targets/features on Rust 1.89"
	@echo "  make test-sanitizer run the FLINT boundary under AddressSanitizer"
	@echo "  make fmt           format tracked Rust and Python sources"
	@echo "  make lint          check Rust/Python/shell quality and docs"
	@echo "  make clean         remove local build and test artifacts"
	@echo
	@echo "FLINT_PREFIX=$(FLINT_PREFIX)"

.PHONY: require-flint
require-flint:
	@if [ -z "$(FLINT_SO)" ]; then \
	  echo "ERROR: FLINT 3.6+ was not found under FLINT_PREFIX=$(FLINT_PREFIX)"; \
	  echo "Set FLINT_PREFIX to an existing installation, or run scripts/install-flint.sh."; \
	  exit 1; \
	fi

.PHONY: require-tools
require-tools:
	@command -v uv >/dev/null || { echo "ERROR: uv is required (https://docs.astral.sh/uv/)"; exit 1; }
	@test -x $(VENV)/bin/maturin || { echo "ERROR: maturin is absent; run 'make sync'"; exit 1; }

.PHONY: require-docs
require-docs:
	@test -x $(VENV)/bin/mkdocs || { echo "ERROR: MkDocs Material is absent; run 'make sync'"; exit 1; }

.PHONY: sync
sync:
	@command -v uv >/dev/null || { echo "ERROR: uv is required (https://docs.astral.sh/uv/)"; exit 1; }
	@VIRTUAL_ENV=$(CURDIR)/$(VENV) uv sync --locked --inexact --no-install-project \
	  --group dev --group validation

.PHONY: engine
engine: engine-portable

define develop-engine
	@rm -f $(CURDIR)/python/noether/_engine*.so \
	  $(CURDIR)/python/noether/_engine*.pyd
	@VIRTUAL_ENV=$(CURDIR)/$(VENV) \
	  $(CURDIR)/$(VENV)/bin/maturin develop --uv --release --skip-install $(1)
	@set -eu; \
	  count=$$(find $(CURDIR)/python/noether -maxdepth 1 \
	    \( -name '_engine*.so' -o -name '_engine*.pyd' \) -print | wc -l | tr -d ' '); \
	  if [ "$$count" -ne 1 ]; then \
	    echo "ERROR: engine build produced $$count in-tree extension artifacts; expected exactly one"; \
	    exit 1; \
	  fi
	@PY=$(PY) scripts/install-native-deps.sh
	@$(MAKE) api-docs
endef

.PHONY: engine-native
engine-native: require-flint require-tools
	@echo "Building the local engine with target-cpu=native ($(RUST_HOST))"
	$(call develop-engine,--config '$(NATIVE_CARGO_CONFIG)')

.PHONY: engine-portable
engine-portable: require-flint require-tools
	$(call develop-engine,)

.PHONY: engine-mpi
engine-mpi: require-flint require-tools
	@echo "Building the local engine with system MPI at the configured CPU baseline"
	$(call develop-engine,--features distributed)

.PHONY: engine-mpi-native
engine-mpi-native: require-flint require-tools
	@echo "Building the local engine with system MPI and target-cpu=native ($(RUST_HOST))"
	$(call develop-engine,--features distributed --config '$(NATIVE_CARGO_CONFIG)')

.PHONY: engine-mpi-portable
engine-mpi-portable: engine-mpi

.PHONY: api-docs
api-docs: require-flint
	@$(PY) scripts/gen_api_doc.py
	@$(PY) scripts/gen_py_api_doc.py
	@$(TEST_PRELOAD) $(PY) scripts/gen_engine_schemas.py
	@$(TEST_PRELOAD) $(PY) scripts/gen_usage_index.py --summary
	@echo "Python API reference: $(CURDIR)/API_REFERENCE.md"

.PHONY: docs docs-rust docs-mpi
docs: require-docs api-docs
	@$(PY) scripts/gen_docs_site.py --output $(CURDIR)/target/noether-docs-src
	@$(VENV)/bin/mkdocs build --strict --config-file $(CURDIR)/target/noether-mkdocs.yml
	@echo "Documentation: $(CURDIR)/target/noether-docs/index.html"

docs-rust: require-flint
	@cargo doc --no-deps -p noether

docs-mpi: require-flint
	@cargo doc --no-deps -p noether --features distributed

.PHONY: wheel
# API generation needs a current extension, but building it with the release
# profile would pay for ThinLTO twice.  The short dev build is only a docs
# bootstrap.  Its Maturin staging directory is discarded before the isolated
# release build because develop-mode RPATH rewriting makes that staging mutable.
wheel: require-flint require-tools
	@set -eu; \
	  mkdir -p "$(CURDIR)/target"; \
	  tmp=$$(mktemp -d "$(CURDIR)/target/.wheel.XXXXXX"); \
	  mkdir -p "$$tmp/dev-engine"; \
	  restore_engine() { \
	    find "$$tmp/dev-engine" -maxdepth 1 -type f \
	      \( -name '_engine*.so' -o -name '_engine*.pyd' \) \
	      -exec cp {} $(CURDIR)/python/noether/ \;; \
	  }; \
	  cleanup() { \
	    if [ -d "$$tmp/dev-engine" ]; then restore_engine; fi; \
	    rm -rf "$$tmp"; \
	  }; \
	  trap cleanup EXIT; \
	  trap 'exit 130' HUP INT TERM; \
	  find $(CURDIR)/python/noether -maxdepth 1 -type f \
	    \( -name '_engine*.so' -o -name '_engine*.pyd' \) \
	    -exec cp {} "$$tmp/dev-engine/" \;; \
	  rm -f $(CURDIR)/python/noether/_engine*.so \
	    $(CURDIR)/python/noether/_engine*.pyd; \
	  TERTIUS_RELOCATABLE=1 VIRTUAL_ENV=$(CURDIR)/$(VENV) \
	    $(CURDIR)/$(VENV)/bin/maturin develop --uv --profile dev \
	      --skip-install --target-dir "$$tmp/target"; \
	  PY=$(PY) $(CURDIR)/scripts/install-native-deps.sh; \
	  $(MAKE) api-docs; \
	  find $(CURDIR)/python/noether -maxdepth 1 -type f \
	    \( -name '_engine*.so' -o -name '_engine*.pyd' \) \
	    -delete; \
	  find "$$tmp/target/maturin" -depth -delete 2>/dev/null || true; \
	  find $(CURDIR)/python/noether -type d -name __pycache__ -prune \
	    -exec rm -rf -- {} +; \
	  find $(CURDIR)/python/noether -type f \
	    \( -name '*.pyc' -o -name '*.pyo' \) -delete; \
	  TERTIUS_RELOCATABLE=1 \
	    $(CURDIR)/$(VENV)/bin/maturin build --release --strip \
	    --auditwheel repair \
	    $(MATURIN_WHEEL_ARGS) \
	    --target-dir "$$tmp/target" \
	    --interpreter "$(CURDIR)/$(PY)" --out "$$tmp"; \
	  wheel=$$(find "$$tmp" -maxdepth 1 -type f -name 'tertius_noether-*.whl' -print); \
	  test $$(printf '%s\n' "$$wheel" | grep -c .) -eq 1 || { \
	    echo "ERROR: expected exactly one tertius-noether wheel"; exit 1; \
	  }; \
	  $(PY) $(CURDIR)/scripts/wheel-native-sbom.py --add "$$wheel"; \
	  $(CURDIR)/scripts/check-wheel.sh "$$wheel"; \
	  mkdir -p "$(WHEEL_OUTPUT)"; \
	  cp "$$wheel" "$(WHEEL_OUTPUT)/"; \
	  echo "wheel: $(WHEEL_OUTPUT)/$$(basename "$$wheel")"

.PHONY: sdist
sdist: require-tools
	@rm -rf $(WHEEL_OUTPUT)/sdist
	@mkdir -p $(WHEEL_OUTPUT)/sdist
	@$(VENV)/bin/maturin sdist --out $(WHEEL_OUTPUT)/sdist
	@archive=$$(find $(WHEEL_OUTPUT)/sdist -maxdepth 1 -name 'tertius_noether-*.tar.gz' -print); \
	  test "$$(printf '%s\n' "$$archive" | grep -c .)" -eq 1; \
	  $(CURDIR)/scripts/check-sdist.sh "$$archive"

.PHONY: check-wheel
check-wheel:
	@test -n "$(WHEEL)" || { echo "usage: make check-wheel WHEEL=path/to/wheel.whl"; exit 2; }
	@scripts/check-wheel.sh "$(WHEEL)"

.PHONY: check-wheel-pythons
check-wheel-pythons:
	@test -n "$(WHEEL)" || { echo "usage: make check-wheel-pythons WHEEL=path/to/wheel.whl"; exit 2; }
	@scripts/check-wheel-pythons.sh "$(WHEEL)"

.PHONY: candidate-linux
candidate-linux:
	@scripts/build-wheel-linux-container.sh

.PHONY: candidate-macos
candidate-macos:
	@set -eu; \
	  if [ "$$(uname -s)" != Darwin ]; then \
	    echo "ERROR: candidate-macos must run natively on macOS"; exit 2; \
	  fi; \
	  arch=$$(uname -m); \
	  case "$$arch" in \
	    arm64) deployment_target=11.0; cpu_policy=arm64 ;; \
	    x86_64) deployment_target=10.13; cpu_policy=x86-64 ;; \
	    *) echo "ERROR: unsupported macOS architecture: $$arch"; exit 2 ;; \
	  esac; \
	  for tool in uv rustc cargo cc make pkg-config shasum otool nm unzip; do \
	    command -v "$$tool" >/dev/null || { \
	      echo "ERROR: $$tool is required; install the macOS build prerequisites first"; \
	      exit 2; \
	    }; \
	  done; \
	  if [ -n "$$(git status --porcelain --untracked-files=all)" ]; then \
	    echo "ERROR: candidate-macos requires a completely clean worktree"; exit 2; \
	  fi; \
	  export MACOSX_DEPLOYMENT_TARGET="$${MACOSX_DEPLOYMENT_TARGET:-$$deployment_target}"; \
	  output="$(CURDIR)/target/release-wheels/macos-$$arch"; \
	  revision=$$(git rev-parse HEAD); \
	  $(MAKE) sync; \
	  cargo fetch --locked; \
	  scripts/install-flint.sh; \
	  rm -rf -- "$$output"; \
	  mkdir -p "$$output"; \
	  scripts/install-flint.sh --copy-sources "$$output/native-sources"; \
	  $(MAKE) wheel WHEEL_DIR="$$output"; \
	  wheel=$$(find "$$output" -maxdepth 1 -type f -name 'tertius_noether-*.whl' -print); \
	  test "$$(printf '%s\n' "$$wheel" | grep -c .)" -eq 1 || { \
	    echo "ERROR: expected exactly one macOS wheel"; exit 1; \
	  }; \
	  scripts/check-wheel.sh "$$wheel" | tee "$$output/wheel-audit.txt"; \
	  scripts/check-wheel-pythons.sh "$$wheel" "$$output/python-metrics"; \
	  shasum -a 256 "$$wheel" >"$$output/SHA256SUMS"; \
	  { \
	    echo "source_revision=$$revision"; \
	    echo "cpu_policy=$$cpu_policy"; \
	    echo "runtime=darwin/$$arch"; \
	    echo "macosx_deployment_target=$$MACOSX_DEPLOYMENT_TARGET"; \
	    echo "[rustc]"; rustc --version --verbose; \
	    echo "[cargo]"; cargo --version; \
	    echo "[uv]"; uv --version; \
	    echo "[cc]"; cc --version | head -n 1; \
	  } >"$$output/build-environment.txt"; \
	  echo "macOS candidate: $$wheel"

.PHONY: image-cpu
image-cpu:
	@scripts/build-images.sh cpu

.PHONY: image-mpi
image-mpi:
	@scripts/build-images.sh mpi

.PHONY: check-image-cpu
check-image-cpu:
	@scripts/check-image.sh cpu $(IMAGE)

.PHONY: check-image-mpi
check-image-mpi:
	@scripts/check-image.sh mpi $(IMAGE)

.PHONY: test
test: require-flint
	@$(TEST_PRELOAD) $(PY) -m pytest \
	  validation/unit/*/test_acceptance_workload.py

.PHONY: test-fast
test-fast: require-flint
	@$(TEST_PRELOAD) $(PY) -m pytest validation/unit \
	  -m "not slow and not benchmark"

.PHONY: test-py
test-py: engine
	@$(TEST_PRELOAD) $(PY) -m pytest validation/unit

.PHONY: test-examples
test-examples: require-flint
	@$(TEST_PRELOAD) $(PY) -m pytest validation/unit/infra/test_examples.py

.PHONY: test-corpus
test-corpus: require-flint
	@$(TEST_PRELOAD) $(PY) -m pytest validation/unit/infra/test_corpus_acceptance.py

.PHONY: test-mpi-representation
test-mpi-representation:
	@scripts/test-m4rst-distributed.sh representation

.PHONY: test-mpi
test-mpi: engine-mpi
	@scripts/test-m4rst-distributed.sh all
	@mpirun --oversubscribe --bind-to none -n 1 $(PY) examples/distributed/f2_linear_algebra.py
	@mpirun --oversubscribe --bind-to none -n 2 $(PY) examples/distributed/f2_linear_algebra.py
	@mpirun --oversubscribe --bind-to none -n 4 $(PY) examples/distributed/f2_linear_algebra.py

.PHONY: coverage-py
coverage-py: require-flint
	@$(TEST_PRELOAD) $(PY) -m pytest validation/unit \
	  --cov=python/noether --cov-branch --cov-report=term-missing:skip-covered \
	  --cov-fail-under=80

.PHONY: coverage-rust
coverage-rust: require-flint
	@cargo llvm-cov --version >/dev/null 2>&1 || { \
	  echo "ERROR: cargo-llvm-cov is required (cargo install cargo-llvm-cov)"; exit 1; \
	}
	@$(CARGO_RELEASE_TEST_ENV) cargo llvm-cov --workspace \
	  --exclude noether-python --exclude m4rst-distributed \
	  --lib --tests --release --features sparb/parallel --summary-only \
	  --fail-under-lines 80

.PHONY: coverage
coverage: coverage-py coverage-rust

.PHONY: test-rust
test-rust: require-flint
	@$(CARGO_RELEASE_TEST_ENV) cargo test --release --workspace --lib --tests \
	  --exclude noether-python --exclude m4rst-distributed \
	  --features sparb/parallel

.PHONY: test-engine
test-engine: require-flint
	@$(CARGO_RELEASE_TEST_ENV) cargo test --release --lib -p noether-python

.PHONY: test-doc
test-doc: require-flint
	@$(CARGO_TEST_ENV) cargo test --doc --workspace \
	  --exclude noether-python --exclude m4rst-distributed \
	  --features sparb/parallel
	@$(CARGO_TEST_ENV) cargo test --doc -p noether-python

.PHONY: test-ignored
test-ignored: require-flint
	@$(CARGO_RELEASE_TEST_ENV) cargo test --release -p noether -- \
	  --include-ignored --skip paper_scale_
	@$(CARGO_RELEASE_TEST_ENV) cargo test --release -p noether-python \
	  -- --include-ignored

.PHONY: test-all
test-all: lint test-rust test-engine test-doc test-py

.PHONY: test-stress
test-stress: test-doc test-ignored
	@PROPTEST_CASES=$${PROPTEST_CASES:-512} $(CARGO_RELEASE_TEST_ENV) cargo test --release --workspace \
	  --lib --tests --exclude noether-python --exclude m4rst-distributed \
	  --features sparb/parallel
	@$(TEST_PRELOAD) $(PY) -m pytest validation/unit

.PHONY: validate
validate: engine-portable
	@$(TEST_PRELOAD) $(PY) -m pytest validation/parity
	@if [ "$${NOETHER_VALIDATE_GAP:-0}" = 1 ]; then \
	  $(TEST_PRELOAD) $(PY) validation/run_parity.py \
	    --topic M1 --require-backend gap; \
	fi
	@if [ "$${NOETHER_VALIDATE_SAGE:-0}" = 1 ]; then \
	  $(TEST_PRELOAD) $(PY) validation/run_parity.py \
	    --topic U1 --require-backend sage; \
	fi

.PHONY: validate-sage
validate-sage: engine-portable
	@NOETHER_VALIDATE_SAGE=1 $(TEST_PRELOAD) $(PY) -m pytest \
	  validation/parity -m sage_oracle
	@NOETHER_VALIDATE_SAGE=1 $(TEST_PRELOAD) $(PY) validation/run_parity.py \
	  --topic U1 --require-backend sage

.PHONY: require-magma
require-magma:
	@$(PY) -c "from validation.harness.cas import resolve_binary; import sys; sys.exit(0 if resolve_binary('magma') else 'ERROR: Magma was not found; set NOETHER_MAGMA_BIN=/path/to/magma')"

.PHONY: validate-magma-release
validate-magma-release: require-magma engine-native
	@mkdir -p $(CURDIR)/target/magma-validation
	@set +e; \
	  scripts/bench-pin.sh $(PY) validation/bench/magma_release.py \
	    --output $(CURDIR)/target/magma-validation/core.json; \
	  core_status=$$?; \
	  if [ $$core_status -eq 1 ]; then \
	    echo "ERROR: core Magma validation failed; surface oracle not started"; \
	    exit 1; \
	  fi; \
	  NOETHER_VALIDATE_MAGMA=1 $(TEST_PRELOAD) $(PY) -m pytest \
	    --require-oracle magma \
	    validation/parity/commalg/test_surface_magma_parity.py \
	    >$(CURDIR)/target/magma-validation/surface.log 2>&1; \
	  surface_status=$$?; \
	  cat $(CURDIR)/target/magma-validation/surface.log; \
	  if [ $$core_status -ne 0 ] || [ $$surface_status -ne 0 ]; then \
	    echo "ERROR: Magma release validation needs resolution"; \
	    exit 1; \
	  fi

.PHONY: check-crates
check-crates:
	@$(PY) scripts/check-crates.py --strict

.PHONY: check-no-gpl
check-no-gpl: require-flint
	@scripts/check-no-gpl.sh release

.PHONY: audit
audit:
	@scripts/audit-dependencies.sh

.PHONY: test-msrv
test-msrv: require-flint
	@$(CARGO_TEST_ENV) scripts/check-msrv.sh

.PHONY: test-sanitizer
test-sanitizer: require-flint
	@$(CARGO_TEST_ENV) scripts/test-sanitizers.sh

.PHONY: fmt
fmt:
	@cargo fmt
	@$(PY) -m ruff format $$(git ls-files '*.py' '*.pyi' | while read -r file; do test ! -f "$$file" || printf '%s\n' "$$file"; done)
	@$(PY) -m ruff check --fix $$(git ls-files '*.py' '*.pyi' | while read -r file; do test ! -f "$$file" || printf '%s\n' "$$file"; done)

.PHONY: fmt-check
fmt-check:
	@cargo fmt --all -- --check
	@$(PY) -m ruff format --check $$(git ls-files '*.py' '*.pyi' | while read -r file; do test ! -f "$$file" || printf '%s\n' "$$file"; done)

.PHONY: python-lint
python-lint:
	@$(PY) -m ruff check $$(git ls-files '*.py' '*.pyi' | while read -r file; do test ! -f "$$file" || printf '%s\n' "$$file"; done)

.PHONY: type-check
type-check:
	@$(PY) -m mypy

.PHONY: shell-check
shell-check:
	@git ls-files '*.sh' | xargs -r -n1 bash -n

.PHONY: doc-check
doc-check:
	@$(PY) scripts/check_doc_freshness.py

.PHONY: check-version
check-version:
	@$(PY) scripts/check-version.py

.PHONY: clippy
clippy:
	@cargo clippy --workspace --all-targets \
	  --exclude m4rst-distributed \
	  --features sparb/parallel -- -D warnings

.PHONY: lint
lint: fmt-check python-lint type-check shell-check doc-check check-version clippy

.PHONY: clean
clean:
	@rm -rf target $(VENV)
	@find crates python validation -type d \
	  \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache -o -name '*.egg-info' \) \
	  -prune -exec rm -rf {} + 2>/dev/null || true
