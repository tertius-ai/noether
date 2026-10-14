# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Unit tests for the object-representation protocol (`noether._repr`).

Pure-Python: no engine, no numpy, no live primitives. Exercises the
`TertiusRepr` mixin and the `harvest` consumer in isolation with dummy
classes. Primitive-level integration (that `Mat`/`Array` expose the protocol)
is covered separately where the engine is available.
"""

from __future__ import annotations

import re
from pathlib import Path

from noether._repr import TertiusRepr, harvest

# ── TertiusRepr.__repr_mimebundle__ ─────────────────────────────────────


def test_bare_type_is_text_plain_only():
    class Bare(TertiusRepr):
        def __repr__(self):
            return "Bare(x=1)"

    b = Bare()
    assert b._repr_mimebundle_() == {"text/plain": "Bare(x=1)"}
    assert b._tertius_summary_() is None


def test_html_shortcut_is_folded_into_bundle():
    class WithHtml(TertiusRepr):
        def __repr__(self):
            return "WithHtml()"

        def _repr_html_(self):
            return "<b>hi</b>"

    assert WithHtml()._repr_mimebundle_() == {
        "text/plain": "WithHtml()",
        "text/html": "<b>hi</b>",
    }


def test_include_exclude_contract():
    class WithHtml(TertiusRepr):
        def __repr__(self):
            return "W()"

        def _repr_html_(self):
            return "<b>hi</b>"

    w = WithHtml()
    assert w._repr_mimebundle_(include={"text/html"}) == {"text/html": "<b>hi</b>"}
    assert "text/html" not in w._repr_mimebundle_(exclude={"text/html"})


def test_raising_shortcut_is_skipped_not_fatal():
    class BadHtml(TertiusRepr):
        def __repr__(self):
            return "BadHtml()"

        def _repr_html_(self):
            raise RuntimeError("boom")

    assert BadHtml()._repr_mimebundle_() == {"text/plain": "BadHtml()"}


def test_empty_shortcut_value_is_omitted():
    class EmptyHtml(TertiusRepr):
        def __repr__(self):
            return "E()"

        def _repr_html_(self):
            return ""

    assert EmptyHtml()._repr_mimebundle_() == {"text/plain": "E()"}


def test_summary_override():
    class CG(TertiusRepr):
        def __repr__(self):
            return "ClassGroup(Z/2Z, h=2)"

        def _tertius_summary_(self):
            return {"invariant_factors": [2], "class_number": 2}

    assert CG()._tertius_summary_() == {"invariant_factors": [2], "class_number": 2}


# ── harvest() ───────────────────────────────────────────────────────────


def test_harvest_collapses_text_plain_only_bundle_to_none():
    """A bundle carrying nothing beyond plain text is not 'rich display';
    harvest reports None so consumers don't double-render `text`."""

    class CG(TertiusRepr):
        def __repr__(self):
            return "ClassGroup(Z/2Z, h=2)"

        def _tertius_summary_(self):
            return {"class_number": 2}

    assert harvest(CG()) == {
        "text": "ClassGroup(Z/2Z, h=2)",
        "type": "CG",
        "mimebundle": None,
        "summary": {"class_number": 2},
    }


def test_harvest_bare_foreign_object():
    class BareForeign:
        def __repr__(self):
            return "BF()"

    h = harvest(BareForeign())
    assert h == {"text": "BF()", "type": "BareForeign", "mimebundle": None, "summary": None}


def test_harvest_foreign_with_html_shortcut_falls_back_to_shortcuts():
    """An object with no `_repr_mimebundle_` (e.g. numpy/matplotlib) still
    yields a bundle assembled from its `_repr_*_` shortcuts."""

    class Foreign:
        def __repr__(self):
            return "Foreign()"

        def _repr_html_(self):
            return "<i>f</i>"

    h = harvest(Foreign())
    assert h["mimebundle"] == {"text/html": "<i>f</i>"}
    assert h["summary"] is None and h["type"] == "Foreign"


def test_harvest_is_bundle_first_authoritative():
    """A type's own `_repr_mimebundle_` wins; the shortcut is NOT separately
    re-probed and merged in."""

    class Bundled:
        def __repr__(self):
            return "Bundled()"

        def _repr_html_(self):
            return "<ignored>"

        def _repr_mimebundle_(self, include=None, exclude=None):
            return {"text/plain": "B", "application/vnd.plotly.v1+json": "{...}"}

    assert harvest(Bundled())["mimebundle"] == {
        "text/plain": "B",
        "application/vnd.plotly.v1+json": "{...}",
    }


def test_harvest_tolerates_data_metadata_tuple_bundle():
    class TupleBundle:
        def __repr__(self):
            return "TB()"

        def _repr_mimebundle_(self, include=None, exclude=None):
            return ({"text/html": "<x>"}, {"text/html": {"meta": 1}})

    assert harvest(TupleBundle())["mimebundle"] == {"text/html": "<x>"}


def test_harvest_tolerates_unreprable_object():
    class Bad:
        def __repr__(self):
            raise ValueError("nope")

    assert harvest(Bad())["text"].startswith("<unreprable Bad")


def test_harvest_ignores_non_dict_summary():
    class WeirdSummary(TertiusRepr):
        def __repr__(self):
            return "WS()"

        def _tertius_summary_(self):
            return "not a dict"  # contract violation -> ignored

    assert harvest(WeirdSummary())["summary"] is None


def test_repr_and_nested_summaries_are_bounded_without_splitting_utf8():
    class Large(TertiusRepr):
        def __repr__(self):
            return "λ" * 10_000

        def _tertius_summary_(self):
            return {
                "text": "μ" * 10_000,
                "values": list(range(100)),
                "nested": {"items": list(range(100))},
            }

    harvested = harvest(Large())

    assert len(harvested["text"].encode("utf-8")) <= 4_096
    assert harvested["text"].endswith("...")
    assert len(harvested["summary"]["text"].encode("utf-8")) <= 4_096
    assert harvested["summary"]["values"][-1] == "..."
    assert harvested["summary"]["nested"]["items"][-1] == "..."


def test_every_owned_pyo3_text_repr_uses_the_shared_bound():
    root = Path(__file__).resolve().parents[3]
    sources = tuple((root / "crates/noether-python/src/py").glob("*.rs"))
    signature = re.compile(r"fn __repr__\(&self\) -> String \{")
    bounded = re.compile(r"fn __repr__\(&self\) -> String \{\n\s+crate::py::repr::bounded_text\(")

    owned_reprs = sum(len(signature.findall(path.read_text())) for path in sources)
    bounded_reprs = sum(len(bounded.findall(path.read_text())) for path in sources)

    assert owned_reprs > 0
    assert bounded_reprs == owned_reprs
