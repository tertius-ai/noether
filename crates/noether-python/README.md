# noether-python

`noether-python` is Noether's private PyO3 transport crate. Maturin compiles it
as `noether._engine`; the ordinary Python package under `python/noether` gives
that extension its public domain layout, type annotations, discovery helpers,
and Python protocols.

The crate depends on the public Rust `noether` facade and translates its types,
results, errors, and accepted discovery index. It does not own competing
mathematical types or algorithms. The Rust facade remains usable as an ordinary
`rlib` without PyO3 or a Python interpreter.

The extension is deliberately private. Users import `noether.linalg`,
`noether.commalg`, and the other public Python domains, never
`noether._engine`. Optional subprocess bridge features are transport adapters
for user-installed tools and are disabled in the conveyed default build.

From the repository root:

```bash
make engine       # build the extension into the source Python package
make test-engine  # Rust-side transport tests
make test-py      # typed Python projection tests
make api-docs     # regenerate checked surface and schema artifacts
```

The wheel uses PyO3's `abi3-py310` surface and carries no Python runtime
dependency. Native-library inclusion and licensing are checked on the final
wheel by `make wheel`.
