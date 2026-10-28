# Noether examples

These 40 executable examples use only the Python standard library and Noether's
public package. They are small enough to read whole, but each performs a real
mathematical calculation in the Rust facade: exact arithmetic, linear algebra,
local and number fields, commutative algebra, finite groups, combinatorics,
higher categories, topology, or geometry.

Run one from the repository root after `make engine`:

```bash
.venv/bin/python examples/10_padic_hensel_lifting.py
```

Run the complete collection with `make test-examples`. The ordinary pytest
suite also executes every file and rejects examples that import a third-party
package other than Noether. `make docs` builds worked examples at
`target/noether-docs/examples/index.html`. Each script's module docstring
introduces the problem and explains the result under `## Result`. The site
extracts imports and calculations from the script, omits the license header
and entry point, and links to the complete source.
