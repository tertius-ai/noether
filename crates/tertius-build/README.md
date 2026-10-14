# tertius-build

`tertius-build` is a private build-dependency containing the shared native-link
mechanics used by Noether crates. It resolves nonempty `*_PREFIX` overrides,
finds the repository-managed FLINT installation, emits link directives, and
sets development or relocatable loader paths consistently.

It is never a runtime dependency and contributes no code to the mathematical
API. Each consuming `build.rs` retains the policy for which native library and
link kind it needs; this crate centralizes only prefix discovery and Cargo link
emission.

```rust,no_run
fn main() {
    tertius_build::link_flint();
}
```

Tests run with the workspace:

```bash
cargo test -p tertius-build
```
