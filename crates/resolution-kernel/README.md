# resolution-kernel

`resolution-kernel` owns deterministic degree-cell scheduling and packed
Schreyer iteration used while constructing free resolutions. Its exact
standard-graded lanes retain certified `groebner-kernel` module frames across
homological degrees. One computes coefficient-field residue ranks and returns
only Betti multiplicities; the other retains every sparse packed differential
for a single projection into the facade. Both use the verified unaugmented
module F4 operation because their consumers do not require change coordinates
back to a noncanonical presentation. Generic exact fields are the correctness
floor; F2, binary-extension, and word-sized prime fields reuse specialized
module row reducers. Arbitrary multigradings remain on the public facade's
exact streamed fallback. The crate has no public-facade or Python dependency.

Full resolution semantics, completeness, minimality, and exact evidence remain
owned by `noether::homalg`. The kernel cancels scalar units in higher packed
frames before the facade projects them into checked module homomorphisms; the
facade handles any cancellation that changes the original presentation. Run
the kernel benchmark with:

```bash
scripts/bench-pin.sh cargo bench -p resolution-kernel --bench betti
```
