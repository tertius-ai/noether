# Noether Python transport — agent guidance

This file refines `../AGENTS.md` for the private PyO3 transport. The public
mathematics is owned by the `noether` Rust crate; this crate translates it.

## Boundary rules

- Do not implement mathematical algorithms here. Add or fix them in the Rust
  facade, then project the resulting type or operation.
- A PyO3 class owns or references the corresponding Rust facade value. It does
  not duplicate algebraic identity, validation, canonicalization, or schema
  truth.
- Convert Rust errors to the narrow public Python exception hierarchy in
  `py/errors.rs`. Do not turn a failed proof or unsupported regime into an
  untyped string result.
- Keep `_engine` private. Public names and domain imports live under
  `python/noether`; users should not need to understand PyO3 registration.
- Keep Python signatures and return values typed. Update the matching `.pyi`
  surface atomically with a new projection and preserve the `py.typed` marker.

Substantial calculations convert dynamic Python arguments to owned facade
values, call `Python::detach`, and construct Python results only after the Rust
work returns. Immutable Rust facade values already stored in a frozen pyclass
are borrowed across the synchronous detach instead of cloned; PyO3's `Ungil`
bound is the compile-time safety gate. Never carry `Bound`, `PyRef`, a Python
callback, or a borrowed Python buffer into the detached closure.
`scripts/gen_api_doc.py` detects direct
detach sites and publishes `releases_gil` in the operation index; keep the
overlap acceptance test green when changing this boundary.

`src/lib.rs` registers the extension surface. `src/py/` contains only adapters,
representation helpers, discovery translation, and optional bridge probes.
The Rust facade's accepted index and schema registry decide which mathematical
types and operations are public.

## Projection changes

A new or changed public operation normally touches:

1. the owning Rust facade type and accepted surface index;
2. its adapter and registration in this crate;
3. the public Python domain module and `.pyi` stub;
4. a Rust/Python surface-projection test and a mathematical acceptance test;
5. generated API/schema artifacts via `make api-docs`.

Generated artifacts are outputs, never input to runtime dispatch. Remote
workers cannot depend on Markdown files.

Optional `*-subprocess` features expose private availability and invocation
adapters from `noether-bridges`. Public mathematical operations own dispatch
and return the same typed result regardless of backend. No linked GPL backend
may enter the conveyed extension.

## Verification

From the repository root:

```bash
make engine
make test-engine
make test-py
make api-docs
```

Run `make test-all` before a PR. A wheel-affecting change also requires
`make wheel`, whose artifact audit—not the source dependency graph—is the final
native/GPL conveyance check.
