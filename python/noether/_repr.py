# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Bounded, local representations for interactive and LLM-facing clients.

``__repr__`` supplies semantic text, ``_repr_mimebundle_`` may add a genuine
rich view, and ``_tertius_summary_`` may provide a lossy JSON-safe summary.
None is a serialization format, and none may trigger remote computation or an
unbounded transfer. Canonical JSON remains the faithful wire representation.
"""

from __future__ import annotations

from itertools import islice
from typing import Any, Iterable, Optional

# Single-purpose IPython-style shortcuts the default bundle harvests, paired
# with the MIME type each maps to. String-valued only — raster/binary forms
# (``_repr_png_`` etc.) return bytes and are left to types that override
# ``_repr_mimebundle_`` directly, so the default bundle stays JSON-clean.
_STRING_SHORTCUTS: tuple[tuple[str, str], ...] = (
    ("_repr_html_", "text/html"),
    ("_repr_latex_", "text/latex"),
    ("_repr_svg_", "image/svg+xml"),
    ("_repr_json_", "application/json"),
)

_TEXT_BYTE_LIMIT = 4_096
_SUMMARY_ITEM_LIMIT = 32
_SUMMARY_DEPTH_LIMIT = 4


class TertiusRepr:
    """Mixin giving a noether type the representation protocol.

    Provides:
      - a default ``_repr_mimebundle_`` that assembles a bundle from
        ``text/plain`` (``repr(self)``) plus any single-purpose ``_repr_*_``
        shortcut the subclass defines — so a type can implement just
        ``_repr_html_`` and still produce a correct bundle;
      - a ``_tertius_summary_`` that defaults to ``None``.

    Override ``_tertius_summary_`` to expose a lossy semantic dict. Override
    ``_repr_mimebundle_`` directly when a type has many rich forms or a form
    with no shortcut (video, glTF). Always provide a good ``__repr__``.
    """

    def _tertius_summary_(self) -> Optional[dict]:
        """A lossy, semantic, JSON-safe dict — or ``None`` if the type has no
        useful summary. NOT a faithful serialization; NOT ``model_dump``."""
        return None

    def _repr_mimebundle_(
        self,
        include: Optional[Iterable[str]] = None,
        exclude: Optional[Iterable[str]] = None,
    ) -> dict:
        """Assemble ``{mimetype: payload}`` from ``__repr__`` plus any
        string-valued ``_repr_*_`` shortcut the subclass defines.

        Honors the IPython ``include`` / ``exclude`` contract. Shortcut calls
        are guarded: a raising or empty shortcut is skipped, never fatal."""
        bundle: dict[str, Any] = {"text/plain": _safe_call_repr(self)}
        for attr, mime in _STRING_SHORTCUTS:
            fn = getattr(self, attr, None)
            if fn is None:
                continue
            try:
                value = fn()
            except Exception:
                continue
            if value:
                bundle[mime] = value

        if include is not None:
            keep = set(include)
            bundle = {k: v for k, v in bundle.items() if k in keep}
        if exclude is not None:
            drop = set(exclude)
            bundle = {k: v for k, v in bundle.items() if k not in drop}
        return bundle


def harvest(obj: Any) -> dict:
    """Read an object's representation into a plain dict of channels.

    The single point where a consumer turns a live object into transportable
    representation data,
    resolving the bundle-first / shortcuts-fallback order so callers do not
    each re-implement it:

      - ``text``      : ``repr(obj)`` (always present).
      - ``type``      : ``type(obj).__name__``.
      - ``mimebundle``: ``obj._repr_mimebundle_()`` if present; else a bundle
                        assembled from the object's ``_repr_*_`` shortcuts.
                        ``None`` if neither exists.
      - ``summary``   : ``obj._tertius_summary_()`` if present, else ``None``.

    Works on any object — noether types, plotly figures, numpy arrays — not
    just ``TertiusRepr`` subclasses, because it probes for the methods rather
    than requiring the mixin. Every probe is guarded.
    """
    out: dict[str, Any] = {
        "text": _safe_call_repr(obj),
        "type": type(obj).__name__,
        "mimebundle": None,
        "summary": None,
    }

    bundle: Optional[dict] = None
    mimebundle_fn = getattr(obj, "_repr_mimebundle_", None)
    if mimebundle_fn is not None:
        try:
            b = mimebundle_fn()
            # IPython allows a (data, metadata) tuple; keep only the data dict.
            if isinstance(b, tuple) and b:
                b = b[0]
            if isinstance(b, dict) and b:
                bundle = b
        except Exception:
            pass
    if bundle is None:
        bundle = _bundle_from_shortcuts(obj) or None
    # A bundle carrying nothing beyond plain text is not "rich display" — that
    # is already in `text`. Report it as None so consumers don't double-render,
    # and so a bare TertiusRepr object matches a bare foreign object.
    if bundle is not None and set(bundle) <= {"text/plain"}:
        bundle = None
    out["mimebundle"] = bundle

    summary_fn = getattr(obj, "_tertius_summary_", None)
    if summary_fn is not None:
        try:
            summary = summary_fn()
            if isinstance(summary, dict):
                out["summary"] = _bounded_summary(summary)
        except Exception:
            pass

    return out


def _safe_call_repr(obj: Any) -> str:
    try:
        return _bounded_text(repr(obj))
    except Exception as exc:  # never let a bad __repr__ break the harvest
        return _bounded_text(f"<unreprable {type(obj).__name__}: {exc}>")


def _bounded_text(value: str) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= _TEXT_BYTE_LIMIT:
        return value
    prefix = encoded[: _TEXT_BYTE_LIMIT - 3]
    while True:
        try:
            return prefix.decode("utf-8") + "..."
        except UnicodeDecodeError:
            prefix = prefix[:-1]


def _bounded_summary(value: Any, depth: int = 0) -> Any:
    if isinstance(value, str):
        return _bounded_text(value)
    if depth >= _SUMMARY_DEPTH_LIMIT:
        return "..."
    if isinstance(value, dict):
        items = list(islice(value.items(), _SUMMARY_ITEM_LIMIT + 1))
        bounded = {
            key: _bounded_summary(item, depth + 1) for key, item in items[:_SUMMARY_ITEM_LIMIT]
        }
        if len(items) > _SUMMARY_ITEM_LIMIT and "_truncated" not in bounded:
            bounded["_truncated"] = True
        return bounded
    if isinstance(value, (list, tuple)):
        items = [_bounded_summary(item, depth + 1) for item in value[:_SUMMARY_ITEM_LIMIT]]
        if len(value) > _SUMMARY_ITEM_LIMIT:
            items.append("...")
        return tuple(items) if isinstance(value, tuple) else items
    return value


def _bundle_from_shortcuts(obj: Any) -> dict:
    """Assemble a mimebundle from an object's string-valued ``_repr_*_``
    shortcuts (no ``text/plain`` — caller already has ``text``). Used as the
    fallback for objects without their own ``_repr_mimebundle_``."""
    bundle: dict[str, Any] = {}
    for attr, mime in _STRING_SHORTCUTS:
        fn = getattr(obj, attr, None)
        if fn is None:
            continue
        try:
            value = fn()
        except Exception:
            continue
        if value:
            bundle[mime] = value
    return bundle
