# noether-bridges guidance

The repository and `crates/AGENTS.md` rules apply. This unpublished crate is
the single production audit point for optional GPL-tool subprocesses.

- Keep one module per executable and one feature per module.
- Link no GPL library and add no in-process FFI here.
- Remove preload and dynamic-library search injection before spawning a tool so
  it uses its own native stack.
- Bound long-running calls with a timeout and avoid pipe-buffer deadlocks.
- Return raw textual output plus typed process errors; mathematical parsing,
  canonicalization and dispatch remain in Noether.
- Every accepted backend has a native correctness floor or a typed unavailable
  result, and a differential validation case when both routes exist.
- The conveyed wheel enables no bridge features. Re-audit it after any change
  to this crate's dependency or feature graph.

Run `cargo test -p noether-bridges --all-features` to exercise the adapters'
public surface. The tests must not require any external executable to be
installed unless explicitly marked as an opt-in oracle test.
