# noether-bridges

`noether-bridges` is Noether's single production audit point for optional,
arm's-length subprocess calls to user-installed mathematical systems. It owns
adapters for msolve, Macaulay2, and Singular.

The crate is unpublished, links no GPL library and has no default features.
Each adapter is enabled separately:

```text
msolve-subprocess
macaulay2-subprocess
singular-subprocess
```

An adapter only probes or spawns its named executable. Mathematical input
encoding, strict output parsing, exact candidate verification, implementation
selection and native fallbacks belong to the Noether Rust facade. A missing
executable is returned as `BridgeError::Unavailable`; it never silently changes
an answer. The process environment removes native preload and library-search
injection so a tool cannot accidentally load Noether's private FLINT/GMP ABI
stack.

The portable wheel profile enables none of these features. They are opt-in capabilities
over tools installed by the user, and the final wheel audit must remain free of
linked GPL libraries and symbols.

```bash
cargo test -p noether-bridges --all-features
make check-no-gpl
```
