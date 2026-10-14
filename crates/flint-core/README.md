# flint-core

`flint-core` supplies safe, owned Rust values over FLINT's raw C ABI. It owns
initialization, clearing, cloning and primitive arithmetic for arbitrary-size
integers and rationals and for real and complex balls. Mathematical parent
contexts, algorithms and dispatch remain in their consuming crates.

Raw declarations come exclusively from upstream `flint3-sys`. The Cargo
dependency is named `flint-sys` internally so downstream source uses the
conventional Rust module name without introducing a second `links = "flint"`
provider.

Noether, `sparla`, `sparb`, `convex-hull`, `geom-core`, and `lattice-enum`
share these values, so exact data does not pass through a second bigint or
rational representation at crate boundaries.

```bash
cargo test -p flint-core --release
cargo doc -p flint-core --no-deps
```

Unsafe code is confined to the four ownership modules named in `AGENTS.md`.
