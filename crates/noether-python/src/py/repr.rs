// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Helpers for the noether object-representation protocol on pyo3 types.
//!
//! The Python-side protocol (`noether._repr`) is `__repr__` (text) +
//! `_repr_mimebundle_` (display) + `_tertius_summary_` (lossy semantic dict).
//! Public engine `#[pyclass]` types implement the same surface.
//! Each type adds the protocol methods to its own `#[pymethods]` block as thin
//! one-liners that delegate to the helpers here.
//!
//! LOCALITY: these helpers take already-rendered strings; they never touch
//! engine-resident data, so a repr/summary call is always cheap and local.
//! The policy is intentionally local-only: representation must never trigger
//! remote computation or an unbounded data transfer.
#![deny(
    clippy::unwrap_used,
    clippy::panic,
    clippy::unreachable,
    clippy::panic_in_result_fn
)]
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

/// Hard byte ceiling for text sent to an interactive or LLM-facing client.
/// The ellipsis is included in the ceiling.
pub(crate) const REPR_BYTE_LIMIT: usize = 4_096;

/// Maximum number of members exposed by one collection-valued summary field.
pub(crate) const SUMMARY_ITEM_LIMIT: usize = 32;

/// Bound one representation without splitting a UTF-8 code point.
pub(crate) fn bounded_text(mut rendered: String) -> String {
    if rendered.len() <= REPR_BYTE_LIMIT {
        return rendered;
    }
    let mut end = REPR_BYTE_LIMIT - 3;
    while !rendered.is_char_boundary(end) {
        end -= 1;
    }
    rendered.truncate(end);
    rendered.push_str("...");
    rendered
}

/// Build an IPython mimebundle dict from a `text/plain` payload plus zero or
/// more `(mimetype, payload)` rich extras (`text/latex`, `image/svg+xml`, …),
/// honoring the include/exclude contract. The single place the bundle/filter
/// logic lives so per-type `_repr_mimebundle_` methods stay one-liners. Empty
/// payloads are skipped.
pub(crate) fn mimebundle(
    py: Python<'_>,
    text: String,
    extras: Vec<(&str, String)>,
    include: Option<Bound<'_, PyAny>>,
    exclude: Option<Bound<'_, PyAny>>,
) -> PyResult<Py<PyDict>> {
    let d = PyDict::new(py);
    d.set_item("text/plain", bounded_text(text))?;
    for (mime, payload) in extras {
        if !payload.is_empty() {
            d.set_item(mime, payload)?;
        }
    }
    if include.is_some() || exclude.is_some() {
        let keys: Vec<String> = d
            .keys()
            .iter()
            .filter_map(|k| k.extract::<String>().ok())
            .collect();
        for k in keys {
            let drop = match (&include, &exclude) {
                (Some(inc), _) if !inc.contains(k.as_str()).unwrap_or(false) => true,
                (_, Some(exc)) if exc.contains(k.as_str()).unwrap_or(false) => true,
                _ => false,
            };
            if drop {
                let _ = d.del_item(k.as_str());
            }
        }
    }
    Ok(d.unbind())
}

#[cfg(test)]
mod tests {
    use super::{bounded_text, REPR_BYTE_LIMIT};

    #[test]
    fn text_bound_is_utf8_safe_and_includes_ellipsis() {
        let rendered = "λ".repeat(REPR_BYTE_LIMIT);
        let bounded = bounded_text(rendered);

        assert!(bounded.len() <= REPR_BYTE_LIMIT);
        assert!(bounded.ends_with("..."));
        assert!(bounded.is_char_boundary(bounded.len()));
        assert_eq!(bounded_text("short".to_owned()), "short");
    }
}
