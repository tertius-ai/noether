# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Generate the public Python operation index from the facade sources.

The Rust facade's accepted type/operation index decides which names exist.
The matching Python package ``__all__`` declarations and adjacent
``__init__.pyi`` files supply projection-specific signatures and schemas,
while ordinary Python and PyO3 docstrings supply descriptions. The result is
an untracked build artifact bundled in wheels; it is never a second
hand-maintained API manifest.

Run from the repository root with::

    python scripts/gen_py_api_doc.py
"""

from __future__ import annotations

import ast
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

NOETHER_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = NOETHER_ROOT / "python"
PACKAGE_ROOT = SRC_ROOT / "noether"
OUTPUT = PACKAGE_ROOT / "API_REFERENCE_PY.json"

sys.path.insert(0, str(NOETHER_ROOT / "scripts"))
import gen_api_doc  # noqa: E402

_PRIMITIVE_SCHEMAS: dict[str, dict[str, Any]] = {
    "Any": {},
    "None": {"type": "null"},
    "bool": {"type": "boolean"},
    "bytes": {"type": "string", "contentEncoding": "base64"},
    "float": {"type": "number"},
    "int": {"type": "integer"},
    "object": {},
    "str": {"type": "string"},
}

# ``ast.TypeAlias`` was added with PEP 695 in Python 3.12.  The package and
# its build tooling support Python 3.10+, whose stubs use the older
# ``Alias: TypeAlias = ...`` spelling.
_TYPE_ALIAS_NODE = getattr(ast, "TypeAlias", None)


def _is_type_alias_node(node: ast.AST) -> bool:
    return _TYPE_ALIAS_NODE is not None and isinstance(node, _TYPE_ALIAS_NODE)


# Sparse editorial metadata belongs beside the generator, not in a generated
# manifest.  Most entries need only their signature and docstring.  These
# representative calculations make every accepted domain discoverable without
# pretending that routine storage routing is a user-facing plan.
_DISCOVERY_METADATA: dict[str, dict[str, Any]] = {
    "linalg.Matrix": {
        "doc": """A dense matrix with entries in one coefficient context.

Construct a matrix with `Matrix.from_rows(ring, rows)`, `Matrix.zeros(ring,
shape)`, or `Matrix.identity(ring, size)`. Rows and columns are indexed from
zero. `A[i, j]` reads one entry; iteration yields rows.

Use `A + B` and `A - B` for addition and subtraction, `A @ B` for matrix
multiplication, and `A * c` for scalar multiplication. Matrix operations
require matching coefficient contexts and compatible dimensions.

`solve`, `kernel`, `image`, `rank`, and `rref` require field coefficients.
Use `solve_exact` for integer equations or congruences, including composite
moduli. Integer, residue-ring, and fixed-precision p-adic normal forms have
separate methods with their own coefficient restrictions.

Matrices are immutable. `to_rows()` materializes a Python list of rows;
accessing one entry does not require exporting the entire matrix.""",
        "related": ("linalg.Matrix.from_rows", "linalg.Vector", "linalg.SparseMatrix"),
    },
    "linalg.Matrix.from_rows": {
        "doc": """Construct a matrix from an iterable of rows over `ring`.

`ring` supplies the coefficient context and converts each entry. All rows
must have the same length. An empty outer iterable constructs a 0-by-0
matrix; use `zeros(ring, (0, n))` to retain a specified column count.

Returns a `Matrix` over the supplied context. Ragged rows raise `ValueError`;
entries that cannot be converted to that context are rejected.""",
        "related": ("linalg.Matrix.zeros", "linalg.Matrix.identity"),
    },
    "linalg.Matrix.zeros": {
        "doc": """Construct the zero matrix over `ring` with shape `(rows, columns)`.

Both dimensions must be nonnegative. Zero dimensions are allowed and
preserved, including `(0, n)` and `(m, 0)`. Returns a `Matrix`.""",
    },
    "linalg.Matrix.identity": {
        "doc": """Construct the `size`-by-`size` identity matrix over `ring`.

`size` must be nonnegative. The size-zero identity has shape `(0, 0)`.
The diagonal entries are the context's multiplicative identity.""",
    },
    "linalg.Matrix.shape": {"summary": "The pair (number of rows, number of columns)."},
    "linalg.Matrix.ring": {"summary": "The coefficient context used to construct this matrix."},
    "linalg.Matrix.input_dimension": {
        "summary": "The number of columns, the dimension of the map's domain."
    },
    "linalg.Matrix.output_dimension": {
        "summary": "The number of rows, the dimension of the map's codomain."
    },
    "linalg.Matrix.__len__": {"summary": "The number of rows."},
    "linalg.Matrix.__iter__": {
        "summary": "Materialize the matrix as Python row lists and iterate over them."
    },
    "linalg.Matrix.__getitem__": {
        "summary": "Read the coefficient at zero-based index (row, column); out-of-range indices raise IndexError."
    },
    "linalg.Matrix.__add__": {
        "summary": "Add matrices with equal shapes and coefficient contexts, entry by entry."
    },
    "linalg.Matrix.__sub__": {
        "summary": "Subtract matrices with equal shapes and coefficient contexts, entry by entry."
    },
    "linalg.Matrix.__neg__": {"summary": "Negate each matrix entry."},
    "linalg.Matrix.__mul__": {
        "summary": "Multiply every matrix entry by a scalar in the coefficient context."
    },
    "linalg.Matrix.__rmul__": {"summary": "Multiply every matrix entry by a scalar on the left."},
    "linalg.Matrix.__matmul__": {
        "summary": "Multiply matrices with matching coefficient contexts and matching inner dimensions."
    },
    "linalg.Matrix.matmul": {
        "summary": "Multiply matrices with matching coefficient contexts and matching inner dimensions.",
        "related": ("linalg.Matrix.apply",),
    },
    "linalg.Matrix.apply": {
        "summary": "Apply the matrix to a vector in the same coefficient context; its length must equal the column count."
    },
    "linalg.Matrix.apply_block": {
        "summary": "Apply the matrix to all columns of a matrix with matching context and inner dimension."
    },
    "linalg.Matrix.to_rows": {"summary": "Materialize all entries as a Python list of row lists."},
    "linalg.Matrix.transpose": {"summary": "Return the transpose, interchanging rows and columns."},
    "linalg.Matrix.solve": {
        "worked_examples": ("03_exact_linear_systems",),
        "doc": """Solve `self @ x = rhs` over a field and return a `SolutionSpace`.

`rhs` must have the same coefficient context as the matrix and length
equal to its row count. The returned affine space records a particular
solution and a basis of homogeneous directions. Every solution is a particular
solution plus a linear combination of those directions.

Use `solve_exact()` for integer or residue-ring coefficients. Context
mismatches raise `ContextMismatchError`; incompatible dimensions raise
`ValueError`. An inconsistent field system raises `ValueError`; it does
not return a `SolutionSpace`.""",
        "related": ("linalg.Matrix.solve_exact", "linalg.Matrix.kernel", "linalg.SolutionSpace"),
    },
    "NoetherError": {"summary": "Base exception for failures reported by Noether."},
    "UnsupportedOperationError": {
        "summary": "The requested operation is unavailable for a mathematical context."
    },
    "ContextMismatchError": {
        "summary": "Values from incompatible mathematical contexts were combined."
    },
    "VerificationError": {"summary": "A mathematical witness failed explicit verification."},
    "Z": {"summary": "The canonical arbitrary-precision integer ring."},
    "Q": {"summary": "The canonical exact rational field."},
    "__version__": {"summary": "The installed Noether package version."},
    "category.SteinerCategory.point": {
        "example": """from noether.category import SteinerCategory
point = SteinerCategory.point()
assert point.dimension == 0
assert point.atom_counts == [1]
point.verify()""",
    },
    "combin.Poset": {
        "example": """from noether.combin import Poset
diamond = Poset(4, [(0, 1), (0, 2), (1, 3), (2, 3)])
assert diamond.leq(0, 3)
assert diamond.covers.edge_count == 4""",
    },
    "combin.Poset.canonical_form": {
        "algorithms": ("principal_downset_canonicalization",),
        "example": """from noether.combin import Poset
poset = Poset(3, [(2, 1), (1, 0)])
result = poset.canonical_form()
assert result.to_json() == '{"order":3,"covers":[[0,1],[1,2]]}'""",
    },
    "commalg.ResidueRing.__call__": {
        "supported_rings": ("Z/nZ",),
        "example": """from noether.commalg import ResidueRing
ring = ResidueRing(15)
result = ring(8) * ring(2)
assert result == ring(1)""",
    },
    "commalg.CrtDecomposition.project": {
        "summary": "Project one residue class to its canonical prime-power CRT components.",
    },
    "linalg.World.initialize": {
        "algorithms": ("system_mpi_world",),
        "backends": ("m4rst-distributed", "m4rst"),
        "example": """from noether import capabilities
installed = capabilities()
if "distributed" in installed.compiled_features:
    from noether.linalg import DistributedMatrix, World
    from noether.numthy import FiniteField
    world = World.initialize()
    matrix = DistributedMatrix.identity(world, FiniteField.prime(2), 3)
    assert matrix.rank() == 3
else:
    assert installed.mpi_runtime is None""",
    },
    "linalg.DistributedError": {"summary": "A distributed Noether operation failed."},
    "commalg.Polynomial.homogeneous_components": {
        "summary": "Decompose a polynomial into components for its ring's positive grading.",
    },
    "commalg.PowerSeries.pow": {
        "summary": "Raise a fixed-truncation series to a nonnegative integer power.",
    },
    "commalg.RationalFunctionField.__call__": {
        "summary": "Coerce a scalar, polynomial, or rational function into this fraction field.",
    },
    "commalg.Scheme.point_count": {
        "summary": "Count finite-field rational points and return a partition certificate.",
        "evidence": ("PointCountCertificate.verify",),
    },
    "geometry.convex_hull": {
        "worked_examples": ("25_exact_geometry",),
        "algorithms": ("exact_rational_convex_hull",),
        "backends": ("convex-hull", "flint"),
        "evidence": ("Polytope.verify",),
        "errors": ("empty point family", "inconsistent ambient dimensions"),
        "example": """from noether.geometry import convex_hull
result = convex_hull([[0, 0], [1, 0], [0, 1]])
result.verify()""",
    },
    "groups.PermutationGroup.from_generators": {
        "worked_examples": ("19_permutation_groups",),
        "algorithms": ("stabilizer_chain",),
        "backends": ("perm-core",),
        "errors": ("degree mismatch", "non-permutation generator"),
        "example": """from noether.groups import Permutation, PermutationGroup
transposition = Permutation.from_images([1, 0, 2])
cycle = Permutation.from_images([1, 2, 0])
result = PermutationGroup.from_generators(3, [transposition, cycle])
assert result.order == 6""",
    },
    "category.SteinerCategory.oriental": {
        "worked_examples": ("28_steiner_categories",),
        "algorithms": ("Street oriental construction",),
        "backends": ("noether",),
        "evidence": ("SteinerCategory.verify",),
        "errors": ("dimension exceeds the bounded construction limit",),
    },
    "homalg.ChainComplex.from_boundaries": {
        "supported_rings": ("Z", "exact fields"),
        "evidence": ("GradedHomology.verify",),
        "example": """from noether import Q
from noether.homalg import ChainComplex
from noether.linalg import Matrix
complex_ = ChainComplex.from_boundaries(Q, {1: Matrix.from_rows(Q, [[0]])})
result = complex_.graded_homology()
result.verify()""",
    },
    "linalg.Matrix.smith_normal_form": {
        "worked_examples": ("04_smith_normal_form",),
        "doc": """Compute the Smith normal form invariant factors of an integer matrix.

Returns `SmithNormalForm`. Its nonzero invariant factors `d_1, ..., d_r`
are positive and satisfy `d_i | d_(i+1)`. For an m-by-n matrix, the cokernel
is the direct sum of the cyclic groups Z/d_i and a free group of rank m-r.
The result retains the input so `verify()` can recheck its claim.
Transformation matrices are available from `smith_normal_form_with_transforms()`.

The default `algorithm="auto"` and `algorithm="exact"` use the unconditional
exact algorithm. `algorithm="smooth"` requires an inclusive `smooth_bound`
and records its smoothness assumption unless the deterministic bound is
fully covered. Inspect the returned validity before using a conditional result.
An invalid smoothness bound raises `ValueError`.""",
        "supported_rings": ("Z",),
        "algorithms": ("exact", "smooth_local"),
        "backends": ("flint", "native"),
        "validity": ("unconditional", "conditional:TorsionOrderIsSmooth"),
        "evidence": ("SmithVerification",),
        "errors": ("invalid smooth bound", "unsupported word precision"),
        "example": """from noether import Z
from noether.linalg import Matrix
result = Matrix.from_rows(Z, [[2, 4], [6, 6]]).smith_normal_form()
assert result.invariant_factors == [2, 6]
result.verify()""",
    },
    "linalg.Matrix.solve_exact": {
        "worked_examples": ("38_exact_linear_systems",),
        "supported_rings": ("Z", "Z/nZ, including composite n"),
        "algorithms": ("Smith normal form", "integer lift and Howell form"),
        "backends": ("flint", "pwvla"),
        "evidence": (
            "IntegerLinearSystemResult.verify",
            "CongruenceSystemResult.verify",
        ),
        "errors": ("dimension mismatch", "incompatible coefficient context"),
    },
    "linalg.Lattice.shortest_vectors": {
        "summary": "Enumerate all shortest signed lattice vectors with an exact certificate.",
        "backends": ("lattice-enum",),
        "evidence": ("LatticeEnumeration.verify",),
    },
    "numthy.factor": {
        "worked_examples": ("07_integer_factorization",),
        "summary": "Return the complete proved factorization of a nonzero integer.",
        "algorithms": ("proved_integer_factorization",),
        "backends": ("flint",),
        "evidence": ("IntegerFactorization.verify",),
        "example": """from noether.numthy import factor
result = factor(-360)
assert result.factors == [(2, 3), (3, 2), (5, 1)]
assert result.verify(-360)""",
    },
    "numthy.partial_factor": {
        "summary": "Return a deterministic partial factorization within explicit work bounds."
    },
    "numthy.PadicRingFrobenius": {
        "summary": "Coefficient Frobenius on a fixed-precision p-adic integer ring."
    },
    "numthy.PadicFieldFrobenius": {
        "summary": "Coefficient Frobenius on a fixed-precision p-adic field."
    },
    "numthy.QadicRingFrobenius": {
        "summary": "Coefficient Frobenius on a fixed-precision unramified Witt ring."
    },
    "numthy.QadicFieldFrobenius": {
        "summary": "Coefficient Frobenius on a fixed-precision unramified field."
    },
    "numthy.FiniteField.extension": {
        "worked_examples": ("08_finite_field_extensions", "09_finite_field_isomorphisms"),
        "summary": "Construct the canonical Conway-style extension field of order p^degree.",
        "supported_rings": ("word-sized prime-power finite fields",),
        "algorithms": ("deterministic Conway presentation",),
        "backends": ("flint",),
        "errors": (
            "nonprime characteristic",
            "zero degree",
            "unavailable Conway polynomial",
        ),
    },
    "commalg.Ideal.groebner_basis": {
        "worked_examples": ("15_groebner_twisted_cubic",),
        "supported_rings": ("polynomial rings over exact admitted coefficient contexts",),
        "algorithms": ("canonical reduced Groebner basis",),
        "backends": ("groebner-kernel",),
        "errors": ("non-polynomial ideal context",),
    },
    "homalg.ChainComplex.homology_invariants": {
        "supported_rings": ("Z", "fixed-precision p-adic integers"),
        "algorithms": ("Smith invariants", "precision-aware p-adic invariants"),
        "backends": ("flint", "sparla"),
        "evidence": (
            "integer: canonical invariants only; use graded_homology for source-retaining evidence",
            "p-adic: PadicModuleInvariants.verify",
        ),
        "errors": ("cochain grading", "incompatible differential contexts"),
    },
    "commalg.QuotientRing.kahler_differentials": {
        "worked_examples": ("39_kahler_differentials",),
        "supported_rings": ("affine algebras over Q", "affine algebras over finite fields"),
        "algorithms": ("exact conormal presentation", "module Groebner normal form"),
        "backends": ("noether", "groebner-kernel", "module-core"),
        "evidence": ("KahlerDifferentials.verify", "ConormalSequenceEvidence.verify"),
        "errors": ("unsupported coefficient context", "invalid quotient presentation"),
    },
    "homalg.FreeModule.exterior_algebra": {
        "worked_examples": ("40_exterior_algebra",),
        "supported_rings": ("supported exact commutative coefficient contexts",),
        "algorithms": ("canonical sparse exterior basis",),
        "backends": ("noether", "module-core"),
        "errors": ("unsupported coefficient context",),
    },
    "numthy.QadicRing.teichmuller": {
        "summary": "Lift a residue-field element to its fixed-precision Teichmuller representative.",
        "backends": ("flint",),
    },
    "topology.SimplicialComplex.graded_homology": {
        "supported_rings": ("Z", "exact fields"),
        "evidence": ("GradedHomology.verify",),
        "example": """from noether.topology import SimplicialComplex
circle = SimplicialComplex([[0, 1], [1, 2], [0, 2]])
result = circle.graded_homology()
assert result.get(1).free_rank == 1
result.verify()""",
    },
    "topology.FilteredComplex.persistent_homology": {
        "worked_examples": ("24_homology_and_persistence",),
        "supported_rings": ("Q", "finite fields"),
        "algorithms": ("specialized prime-field reduction", "generic exact reduction"),
        "backends": ("m4rst", "m4rste", "noether generic"),
        "errors": ("unsupported coefficient context", "integral barcode request"),
    },
}


def _accepted_surface() -> tuple[
    tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]
]:
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))
    from noether._engine_gateway import _import_engine

    domains, types, operations, values = _import_engine()._accepted_surface()
    return tuple(domains), tuple(types), tuple(operations), tuple(values)


def _public_modules() -> tuple[str, ...]:
    domains, _, _, _ = _accepted_surface()
    return ("noether", *(f"noether.{domain}" for domain in domains))


def _entry_category(entry: dict[str, Any]) -> str:
    if entry["kind"] in {"class", "type_alias"}:
        return "types"
    if entry["kind"] == "value":
        return "values"
    return "operations"


def _validate_rust_acceptance(entries: list[dict[str, Any]]) -> None:
    _, accepted_types, accepted_operations, accepted_values = _accepted_surface()
    accepted = {
        "types": set(accepted_types),
        "operations": set(accepted_operations),
        "values": set(accepted_values),
    }
    projected = {
        category: {
            entry["canonical_name"] for entry in entries if _entry_category(entry) == category
        }
        for category in ("types", "operations", "values")
    }
    if projected == accepted:
        return
    details = []
    for category in ("types", "operations", "values"):
        missing = sorted(accepted[category] - projected[category])
        unexpected = sorted(projected[category] - accepted[category])
        if missing:
            details.append(f"missing Rust-accepted {category}: {', '.join(missing[:8])}")
        if unexpected:
            details.append(f"unaccepted Python {category}: {', '.join(unexpected[:8])}")
    raise RuntimeError("Python projection diverges from Rust: " + "; ".join(details))


def _paths(module: str) -> tuple[Path, Path]:
    relative = module.removeprefix("noether").lstrip(".").replace(".", "/")
    directory = PACKAGE_ROOT / relative if relative else PACKAGE_ROOT
    return directory / "__init__.py", directory / "__init__.pyi"


def _parse(path: Path) -> ast.Module:
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as error:
        raise RuntimeError(f"cannot parse public API source {path}: {error}") from error


def _unparse(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def _literal_strings(node: ast.AST) -> list[str] | None:
    if not isinstance(node, (ast.List, ast.Tuple)):
        return None
    values: list[str] = []
    for element in node.elts:
        if not isinstance(element, ast.Constant) or not isinstance(element.value, str):
            return None
        values.append(element.value)
    return values


def _declared_exports(tree: ast.Module, path: Path) -> list[str]:
    exports: list[str] | None = None
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == "__all__" for target in targets):
            continue
        values = _literal_strings(node.value) if node.value is not None else None
        if values is None:
            raise RuntimeError(f"{path}: __all__ must be a literal list of strings")
        exports = values
        break
    if exports is None:
        raise RuntimeError(f"{path}: public facade has no __all__ declaration")
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.AugAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "__all__"
            and isinstance(node.op, ast.Add)
        ):
            continue
        values = _literal_strings(node.value)
        if values is None:
            raise RuntimeError(f"{path}: conditional __all__ additions must be literal strings")
        exports.extend(values)
    if len(exports) != len(set(exports)):
        raise RuntimeError(f"{path}: __all__ contains duplicate names")
    return exports


def _engine_targets(tree: ast.Module) -> dict[str, str]:
    """Map a facade name to the private PyO3 name it projects."""

    targets: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
            continue
        names = node.targets if isinstance(node, ast.Assign) else [node.target]
        public_names = [target.id for target in names if isinstance(target, ast.Name)]
        value = node.value
        if isinstance(value, ast.Attribute):
            base = value.value
            if isinstance(base, ast.Name) and base.id == "_engine":
                target = value.attr
            elif (
                isinstance(base, ast.Call)
                and isinstance(base.func, ast.Name)
                and base.func.id == "_import_engine"
            ):
                target = value.attr
            else:
                continue
            for public_name in public_names:
                targets[public_name] = target
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        delegated = {
            call.func.attr
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "_engine"
        }
        if len(delegated) == 1:
            targets[node.name] = delegated.pop()
    return targets


def _decorators(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    result: set[str] = set()
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Name):
            result.add(decorator.id)
        elif isinstance(decorator, ast.Attribute):
            result.add(decorator.attr)
    return result


def _method_kind(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    decorators = _decorators(node)
    if "classmethod" in decorators:
        return "classmethod"
    if "staticmethod" in decorators:
        return "staticmethod"
    if "property" in decorators:
        return "property"
    if isinstance(node, ast.AsyncFunctionDef):
        return "async_method"
    return "method"


def _format_arg(arg: ast.arg) -> str:
    annotation = f": {_unparse(arg.annotation)}" if arg.annotation is not None else ""
    return f"{arg.arg}{annotation}"


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = node.args
    positional = [*args.posonlyargs, *args.args]
    default_offset = len(positional) - len(args.defaults)
    parts: list[str] = []
    for index, arg in enumerate(positional):
        part = _format_arg(arg)
        if index >= default_offset:
            part += f"={_unparse(args.defaults[index - default_offset])}"
        parts.append(part)
        if args.posonlyargs and index == len(args.posonlyargs) - 1:
            parts.append("/")
    if args.vararg is not None:
        parts.append(f"*{_format_arg(args.vararg)}")
    elif args.kwonlyargs:
        parts.append("*")
    for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
        part = _format_arg(arg)
        if default is not None:
            part += f"={_unparse(default)}"
        parts.append(part)
    if args.kwarg is not None:
        parts.append(f"**{_format_arg(args.kwarg)}")
    result = f"({', '.join(parts)})"
    if node.returns is not None:
        result += f" -> {_unparse(node.returns)}"
    return result


def _annotation_schema(node: ast.AST | None) -> dict[str, Any]:
    if node is None:
        return {}
    text = _unparse(node)
    if isinstance(node, ast.Constant) and node.value is None:
        return {"type": "null"}
    if isinstance(node, ast.Name):
        primitive = _PRIMITIVE_SCHEMAS.get(node.id)
        return dict(primitive) if primitive is not None else {"x-python-type": node.id}
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        members: list[dict[str, Any]] = []

        def append_union(part: ast.AST) -> None:
            if isinstance(part, ast.BinOp) and isinstance(part.op, ast.BitOr):
                append_union(part.left)
                append_union(part.right)
            else:
                members.append(_annotation_schema(part))

        append_union(node)
        return {"anyOf": members, "x-python-type": text}
    if isinstance(node, ast.Subscript):
        base = _unparse(node.value)
        arguments = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
        if base in {"list", "set", "frozenset", "Sequence", "Iterable", "Iterator"}:
            item = _annotation_schema(arguments[0]) if arguments else {}
            return {"type": "array", "items": item, "x-python-type": text}
        if base == "tuple":
            if (
                len(arguments) == 2
                and isinstance(arguments[1], ast.Constant)
                and arguments[1].value is Ellipsis
            ):
                return {
                    "type": "array",
                    "items": _annotation_schema(arguments[0]),
                    "x-python-type": text,
                }
            return {
                "type": "array",
                "prefixItems": [_annotation_schema(argument) for argument in arguments],
                "minItems": len(arguments),
                "maxItems": len(arguments),
                "x-python-type": text,
            }
        if base in {"dict", "Mapping"}:
            value = _annotation_schema(arguments[1]) if len(arguments) > 1 else {}
            return {"type": "object", "additionalProperties": value, "x-python-type": text}
        if base == "Literal":
            values = [ast.literal_eval(argument) for argument in arguments]
            return {"enum": values, "x-python-type": text}
        return {"x-python-type": text}
    return {"x-python-type": text}


def _parameter_records(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[dict[str, Any]]:
    args = node.args
    positional = [*args.posonlyargs, *args.args]
    default_offset = len(positional) - len(args.defaults)
    records: list[dict[str, Any]] = []
    for index, arg in enumerate(positional):
        if arg.arg in {"self", "cls"}:
            continue
        default = args.defaults[index - default_offset] if index >= default_offset else None
        record: dict[str, Any] = {
            "name": arg.arg,
            "kind": "positional_only" if index < len(args.posonlyargs) else "positional_or_keyword",
            "required": default is None,
            "schema": _annotation_schema(arg.annotation),
        }
        if default is not None:
            record["default"] = _unparse(default)
        records.append(record)
    if args.vararg is not None:
        records.append(
            {
                "name": args.vararg.arg,
                "kind": "var_positional",
                "required": False,
                "schema": {"type": "array", "items": _annotation_schema(args.vararg.annotation)},
            }
        )
    for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
        record = {
            "name": arg.arg,
            "kind": "keyword_only",
            "required": default is None,
            "schema": _annotation_schema(arg.annotation),
        }
        if default is not None:
            record["default"] = _unparse(default)
        records.append(record)
    if args.kwarg is not None:
        records.append(
            {
                "name": args.kwarg.arg,
                "kind": "var_keyword",
                "required": False,
                "schema": {
                    "type": "object",
                    "additionalProperties": _annotation_schema(args.kwarg.annotation),
                },
            }
        )
    return records


def _input_schema(parameters: Iterable[dict[str, Any]]) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    additional = False
    for parameter in parameters:
        if parameter["kind"] == "var_keyword":
            additional = parameter["schema"].get("additionalProperties", {})
            continue
        properties[parameter["name"]] = parameter["schema"]
        if parameter["required"] and parameter["kind"] != "var_positional":
            required.append(parameter["name"])
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": additional,
    }
    if required:
        schema["required"] = required
    return schema


def _overload(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, Any]:
    parameters = _parameter_records(node)
    return {
        "signature": _signature(node),
        "parameters": parameters,
        "input_schema": _input_schema(parameters),
        "result_schema": _annotation_schema(node.returns),
    }


def _summary(doc: str) -> str:
    paragraph = doc.strip().split("\n\n", 1)[0].replace("\n", " ").strip()
    if not paragraph:
        return ""
    end = paragraph.find(". ")
    return paragraph if end < 0 else paragraph[: end + 1]


def _engine_docs() -> tuple[dict[str, str], dict[str, str], set[str]]:
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))
    from noether._engine_gateway import _import_engine

    engine = _import_engine()
    docs: dict[str, str] = {}
    type_ids: dict[str, str] = {}
    gil_releasing: set[str] = set()
    for cls in gen_api_doc.collect_pyclasses():
        name = cls["py_name"]
        runtime_class = getattr(engine, name, None)
        runtime_class_doc = getattr(runtime_class, "__doc__", None)
        extracted_class_doc = cls.get("doc", "")
        docs[name] = (
            runtime_class_doc
            if extracted_class_doc
            and isinstance(runtime_class_doc, str)
            and runtime_class_doc.strip()
            else extracted_class_doc
        )
        runtime_type_id = getattr(runtime_class, "type_id", None)
        if isinstance(runtime_type_id, str):
            type_ids[name] = runtime_type_id
        for method in cls["methods"]:
            method_name = method["py_name"]
            runtime_method = getattr(runtime_class, method_name, None)
            runtime_method_doc = getattr(runtime_method, "__doc__", None)
            extracted_method_doc = method.get("doc", "")
            docs[f"{name}.{method_name}"] = (
                runtime_method_doc
                if extracted_method_doc
                and isinstance(runtime_method_doc, str)
                and runtime_method_doc.strip()
                else extracted_method_doc
            )
            if method.get("releases_gil", False):
                gil_releasing.add(f"{name}.{method_name}")
    for function in gen_api_doc.collect_pyfunctions():
        name = function["py_name"]
        runtime_function = getattr(engine, name, None)
        runtime_function_doc = getattr(runtime_function, "__doc__", None)
        extracted_function_doc = function.get("doc", "")
        docs[name] = (
            runtime_function_doc
            if extracted_function_doc
            and isinstance(runtime_function_doc, str)
            and runtime_function_doc.strip()
            else extracted_function_doc
        )
        if function.get("releases_gil", False):
            gil_releasing.add(name)
    return docs, type_ids, gil_releasing


def _nodes_by_name(tree: ast.Module) -> dict[str, list[ast.AST]]:
    result: dict[str, list[ast.AST]] = {}
    for node in tree.body:
        name: str | None = None
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
        elif isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            for assigned_name in names:
                result.setdefault(assigned_name, []).append(node)
        elif _is_type_alias_node(node):
            alias_name = getattr(node, "name", None)
            if isinstance(alias_name, ast.Name):
                name = alias_name.id
        if name is not None:
            result.setdefault(name, []).append(node)
    return result


def _runtime_docs(tree: ast.Module) -> dict[str, str]:
    docs: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            docs[node.name] = ast.get_docstring(node) or ""
        if isinstance(node, ast.ClassDef):
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    docs[f"{node.name}.{member.name}"] = ast.get_docstring(member) or ""
    return docs


def _imported_runtime_docs(tree: ast.Module) -> dict[str, str]:
    """Follow direct package imports for public definitions kept in helpers."""

    docs: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        if not node.module.startswith("noether."):
            continue
        imported_path = SRC_ROOT.joinpath(*node.module.split(".")).with_suffix(".py")
        if not imported_path.is_file():
            continue
        imported_docs = _runtime_docs(_parse(imported_path))
        for alias in node.names:
            public_name = alias.asname or alias.name
            if alias.name in imported_docs:
                docs[public_name] = imported_docs[alias.name]
            prefix = f"{alias.name}."
            for qualified_name, document in imported_docs.items():
                if qualified_name.startswith(prefix):
                    docs[f"{public_name}.{qualified_name.removeprefix(prefix)}"] = document
    return docs


def _canonical_name(module: str, qualified_name: str) -> str:
    domain = module.removeprefix("noether").lstrip(".")
    return f"{domain}.{qualified_name}" if domain else qualified_name


def _base_entry(
    *,
    module: str,
    name: str,
    qualified_name: str,
    kind: str,
    doc: str,
    file: Path,
    line: int,
    overloads: list[dict[str, Any]] | None = None,
    type_id: str | None = None,
    releases_gil: bool = False,
) -> dict[str, Any]:
    overloads = overloads or []
    canonical_name = _canonical_name(module, qualified_name)
    entry: dict[str, Any] = {
        "name": name,
        "qualified_name": qualified_name,
        "canonical_name": canonical_name,
        "module": module,
        "domain": module.removeprefix("noether.") if module != "noether" else "root",
        "kind": kind,
        "signature": overloads[0]["signature"] if overloads else "",
        "signatures": [overload["signature"] for overload in overloads],
        "summary": _summary(doc),
        "doc": doc,
        "overloads": overloads,
        "input_schema": overloads[0]["input_schema"] if overloads else {},
        "result_schema": overloads[0]["result_schema"] if overloads else {},
        "determinism": (
            "seeded"
            if any(
                parameter["name"] == "seed"
                for overload in overloads
                for parameter in overload["parameters"]
            )
            else "deterministic"
        ),
        "file": str(file.relative_to(SRC_ROOT)),
        "line": line,
        "releases_gil": releases_gil,
    }
    if type_id is not None:
        entry["type_id"] = type_id
    entry.update(_DISCOVERY_METADATA.get(canonical_name, {}))
    metadata = _DISCOVERY_METADATA.get(canonical_name, {})
    if "doc" in metadata and "summary" not in metadata:
        entry["summary"] = _summary(entry["doc"])
    if not entry["summary"]:
        entry["summary_source"] = "missing"
    else:
        entry["summary_source"] = (
            "editorial metadata"
            if "summary" in metadata or "doc" in metadata
            else "public docstring"
        )
    return entry


def _function_groups(nodes: Iterable[ast.AST]) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [node for node in nodes if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]


def _class_entries(
    *,
    module: str,
    class_node: ast.ClassDef,
    runtime_docs: dict[str, str],
    engine_target: str | None,
    engine_docs: dict[str, str],
    gil_releasing: set[str],
    type_ids: dict[str, str],
    stub_path: Path,
) -> list[dict[str, Any]]:
    public_name = class_node.name
    engine_name = engine_target or public_name
    class_doc = runtime_docs.get(public_name) or engine_docs.get(engine_name, "")
    grouped: dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]] = {}
    for member in class_node.body:
        if not isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if member.name.startswith("_") and not (
            member.name.startswith("__") and member.name.endswith("__")
        ):
            continue
        grouped.setdefault(member.name, []).append(member)

    constructors = grouped.pop("__init__", [])
    entries = [
        _base_entry(
            module=module,
            name=public_name,
            qualified_name=public_name,
            kind="class",
            doc=class_doc,
            file=stub_path,
            line=class_node.lineno,
            overloads=[_overload(node) for node in constructors],
            type_id=type_ids.get(engine_name),
        )
    ]
    for method_name, methods in grouped.items():
        document = runtime_docs.get(f"{public_name}.{method_name}") or engine_docs.get(
            f"{engine_name}.{method_name}", ""
        )
        entries.append(
            _base_entry(
                module=module,
                name=method_name,
                qualified_name=f"{public_name}.{method_name}",
                kind=_method_kind(methods[0]),
                doc=document,
                file=stub_path,
                line=methods[0].lineno,
                overloads=[_overload(node) for node in methods],
                type_id=type_ids.get(engine_name),
                releases_gil=f"{engine_name}.{method_name}" in gil_releasing,
            )
        )
    return entries


def _module_entries(
    module: str,
    engine_docs: dict[str, str],
    type_ids: dict[str, str],
    gil_releasing: set[str],
) -> list[dict[str, Any]]:
    source_path, stub_path = _paths(module)
    source_tree = _parse(source_path)
    stub_tree = _parse(stub_path)
    exports = _declared_exports(source_tree, source_path)
    targets = _engine_targets(source_tree)
    declarations = _nodes_by_name(stub_tree)
    docs = _imported_runtime_docs(source_tree)
    docs.update(_runtime_docs(source_tree))
    entries: list[dict[str, Any]] = []

    missing = [name for name in exports if name not in declarations]
    if missing:
        raise RuntimeError(
            f"{stub_path}: declared exports have no typed declaration: {', '.join(missing)}"
        )

    for name in exports:
        nodes = declarations[name]
        class_nodes = [node for node in nodes if isinstance(node, ast.ClassDef)]
        functions = _function_groups(nodes)
        if class_nodes:
            entries.extend(
                _class_entries(
                    module=module,
                    class_node=class_nodes[0],
                    runtime_docs=docs,
                    engine_target=targets.get(name),
                    engine_docs=engine_docs,
                    gil_releasing=gil_releasing,
                    type_ids=type_ids,
                    stub_path=stub_path,
                )
            )
            continue
        if functions:
            engine_name = targets.get(name, name)
            document = docs.get(name) or engine_docs.get(engine_name, "")
            entries.append(
                _base_entry(
                    module=module,
                    name=name,
                    qualified_name=name,
                    kind="async_function"
                    if isinstance(functions[0], ast.AsyncFunctionDef)
                    else "function",
                    doc=document,
                    file=stub_path,
                    line=functions[0].lineno,
                    overloads=[_overload(node) for node in functions],
                    releases_gil=engine_name in gil_releasing,
                )
            )
            continue

        declaration = nodes[0]
        if _is_type_alias_node(declaration):
            annotation = getattr(declaration, "value", None)
            kind = "type_alias"
        else:
            annotation = declaration.annotation if isinstance(declaration, ast.AnnAssign) else None
            is_type_alias = annotation is not None and _unparse(annotation) == "TypeAlias"
            kind = "type_alias" if is_type_alias else "value"
            if is_type_alias and isinstance(declaration, ast.AnnAssign):
                annotation = declaration.value
        entries.append(
            _base_entry(
                module=module,
                name=name,
                qualified_name=name,
                kind=kind,
                doc=docs.get(name, ""),
                file=stub_path,
                line=getattr(declaration, "lineno", 0),
                overloads=[],
                type_id=type_ids.get(targets.get(name, "")),
            )
        )
        entries[-1]["result_schema"] = _annotation_schema(annotation)
        declaration = _unparse(annotation)
        if kind == "type_alias":
            entries[-1]["python_declaration"] = f"type {name} = {declaration}"
        else:
            entries[-1]["python_declaration"] = f"{name}: {declaration}" if declaration else name
    return entries


def build_index() -> list[dict[str, Any]]:
    engine_docs, type_ids, gil_releasing = _engine_docs()
    entries: list[dict[str, Any]] = []
    for module in _public_modules():
        entries.extend(_module_entries(module, engine_docs, type_ids, gil_releasing))
    entries.sort(key=lambda entry: entry["canonical_name"])
    canonical_names = [entry["canonical_name"] for entry in entries]
    duplicates = sorted({name for name in canonical_names if canonical_names.count(name) > 1})
    if duplicates:
        raise RuntimeError(f"duplicate public operation descriptions: {', '.join(duplicates)}")
    unknown_metadata = sorted(set(_DISCOVERY_METADATA) - set(canonical_names))
    if unknown_metadata:
        raise RuntimeError(
            "discovery metadata names unaccepted operations: " + ", ".join(unknown_metadata)
        )
    _validate_rust_acceptance(entries)
    return entries


def render_json(entries: list[dict[str, Any]]) -> str:
    return json.dumps(entries, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def main() -> int:
    try:
        entries = build_index()
    except RuntimeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    OUTPUT.write_text(render_json(entries), encoding="utf-8")
    missing = [entry["canonical_name"] for entry in entries if not entry["summary"]]
    if missing:
        print(
            f"Missing authored descriptions: {len(missing)}; "
            "inspect entries with summary_source='missing' in the generated index.",
            file=sys.stderr,
        )
    domains = {entry["domain"] for entry in entries}
    operations = sum(entry["kind"] not in {"class", "type_alias", "value"} for entry in entries)
    print(
        f"Wrote {OUTPUT.relative_to(NOETHER_ROOT)}: {len(entries)} declared entries "
        f"({operations} operations) across {len(domains)} public domains"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
