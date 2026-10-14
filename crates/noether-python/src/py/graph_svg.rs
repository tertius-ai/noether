// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Static node-link SVG rendering for graph pyclasses (noether._repr display).
//!
//! Emitted as `image/svg+xml` in `Graph._repr_mimebundle_`. Self-contained (no
//! JS), so it renders in any frontend. SIZE-GATED and BOUNDED: only graphs up
//! to `MAX_SVG_NODES` are drawn (above that the caller falls back to a text
//! summary), and the layout runs a fixed iteration count — a repr stays cheap
//! and local, never an unbounded compute.
#![deny(
    clippy::unwrap_used,
    clippy::panic,
    clippy::unreachable,
    clippy::panic_in_result_fn
)]
use std::fmt::Write as _;

/// Largest graph we draw inline. Above this, render nothing (summary fallback).
pub(crate) const MAX_SVG_NODES: usize = 150;

const ITERS: usize = 120;
const VIEW: f64 = 420.0;
const MARGIN: f64 = 24.0;

/// Fruchterman–Reingold layout in the unit square. Deterministic: circular
/// initialisation (no RNG) refined by a fixed number of cooling iterations.
fn fr_layout(n: usize, edges: &[(usize, usize)]) -> Vec<(f64, f64)> {
    use std::f64::consts::TAU;
    let mut pos: Vec<(f64, f64)> = (0..n)
        .map(|i| {
            let a = TAU * (i as f64) / (n.max(1) as f64);
            (0.5 + 0.4 * a.cos(), 0.5 + 0.4 * a.sin())
        })
        .collect();
    if n <= 1 {
        return pos;
    }
    let k = (1.0 / n as f64).sqrt(); // ideal edge length on a unit area
    let mut temp = 0.10_f64;
    let cool = temp / (ITERS as f64 + 1.0);
    let mut disp = vec![(0.0_f64, 0.0_f64); n];
    for _ in 0..ITERS {
        for d in disp.iter_mut() {
            *d = (0.0, 0.0);
        }
        // Repulsion between every pair (O(n^2); fine for n <= MAX_SVG_NODES).
        for i in 0..n {
            for j in (i + 1)..n {
                let dx = pos[i].0 - pos[j].0;
                let dy = pos[i].1 - pos[j].1;
                let dist = (dx * dx + dy * dy).sqrt().max(1e-4);
                let f = k * k / dist;
                let (ux, uy) = (dx / dist, dy / dist);
                disp[i].0 += ux * f;
                disp[i].1 += uy * f;
                disp[j].0 -= ux * f;
                disp[j].1 -= uy * f;
            }
        }
        // Attraction along edges.
        for &(a, b) in edges {
            if a == b || a >= n || b >= n {
                continue;
            }
            let dx = pos[a].0 - pos[b].0;
            let dy = pos[a].1 - pos[b].1;
            let dist = (dx * dx + dy * dy).sqrt().max(1e-4);
            let f = dist * dist / k;
            let (ux, uy) = (dx / dist, dy / dist);
            disp[a].0 -= ux * f;
            disp[a].1 -= uy * f;
            disp[b].0 += ux * f;
            disp[b].1 += uy * f;
        }
        // Apply, capped by temperature, clamped to the unit square.
        for i in 0..n {
            let (dx, dy) = disp[i];
            let d = (dx * dx + dy * dy).sqrt().max(1e-4);
            let lim = d.min(temp);
            pos[i].0 = (pos[i].0 + dx / d * lim).clamp(0.0, 1.0);
            pos[i].1 = (pos[i].1 + dy / d * lim).clamp(0.0, 1.0);
        }
        temp = (temp - cool).max(cool);
    }
    pos
}

/// Render an undirected/directed node-link diagram as an SVG string, or `None`
/// if the graph is empty or larger than `MAX_SVG_NODES` (caller falls back to a
/// summary). `edges` is the raw edge list; undirected duplicates are merged.
pub(crate) fn node_link_svg(
    n: usize,
    edges: Vec<(usize, usize)>,
    directed: bool,
) -> Option<String> {
    node_link_svg_colored(n, edges, directed, &vec![None; n])
}

/// Same as [`node_link_svg`], but `node_colors[i]` (an SVG color, e.g.
/// `"#16a34a"`) overrides node `i`'s default blue fill when `Some`. Used by
/// the self-contained result-type overlays ([`shortest_path_svg`],
/// [`traversal_tree_svg`]) to mark source/target vertices without needing
/// the full source graph.
pub(crate) fn node_link_svg_colored(
    n: usize,
    mut edges: Vec<(usize, usize)>,
    directed: bool,
    node_colors: &[Option<&str>],
) -> Option<String> {
    if n == 0 || n > MAX_SVG_NODES {
        return None;
    }
    if !directed {
        for e in edges.iter_mut() {
            if e.0 > e.1 {
                *e = (e.1, e.0);
            }
        }
        edges.sort_unstable();
        edges.dedup();
    }
    let pos = fr_layout(n, &edges);
    let span = VIEW - 2.0 * MARGIN;
    let map = |p: (f64, f64)| (MARGIN + p.0 * span, MARGIN + p.1 * span);

    let mut s = String::new();
    let _ = write!(
        s,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{VIEW}\" height=\"{VIEW}\" \
         viewBox=\"0 0 {VIEW} {VIEW}\">"
    );
    for &(a, b) in &edges {
        if a >= n || b >= n {
            continue;
        }
        let (x1, y1) = map(pos[a]);
        let (x2, y2) = map(pos[b]);
        let _ = write!(
            s,
            "<line x1=\"{x1:.1}\" y1=\"{y1:.1}\" x2=\"{x2:.1}\" y2=\"{y2:.1}\" \
             stroke=\"#9ca3af\" stroke-width=\"1\"/>"
        );
    }
    for (i, &position) in pos.iter().enumerate().take(n) {
        let (x, y) = map(position);
        let fill = node_colors.get(i).copied().flatten().unwrap_or("#3b82f6");
        let _ = write!(
            s,
            "<circle cx=\"{x:.1}\" cy=\"{y:.1}\" r=\"5\" fill=\"{fill}\" \
             stroke=\"#1e3a8a\" stroke-width=\"1\"/>"
        );
    }
    let _ = write!(s, "</svg>");
    Some(s)
}

/// Shared core for the "highlight a subset within the ambient graph" family
/// ([`shortest_path_svg`], [`traversal_tree_svg`]): every vertex/edge of the
/// *full* graph is laid out and drawn faint gray first, then
/// `highlight_edges` are redrawn bold and `node_colors[i]` (`Some`) redraws
/// vertex `i` in that color — so a path or traversal tree renders *in
/// context*, not as an isolated diagram.
///
/// `highlight_edges` must use the same endpoint order the caller's own
/// highlighted edges naturally have; normalized internally to match how the
/// background `edges` get normalized for an undirected graph, so a
/// highlighted edge is correctly recognized as "don't also draw it faint."
fn node_link_svg_highlighted(
    n: usize,
    mut edges: Vec<(usize, usize)>,
    directed: bool,
    node_colors: &[Option<&str>],
    highlight_edges: std::collections::HashSet<(usize, usize)>,
) -> Option<String> {
    if n == 0 || n > MAX_SVG_NODES {
        return None;
    }
    if !directed {
        for e in edges.iter_mut() {
            if e.0 > e.1 {
                *e = (e.1, e.0);
            }
        }
        edges.sort_unstable();
        edges.dedup();
    }
    let highlight_edges: std::collections::HashSet<(usize, usize)> = if directed {
        highlight_edges
    } else {
        highlight_edges
            .into_iter()
            .map(|(a, b)| if a > b { (b, a) } else { (a, b) })
            .collect()
    };
    let pos = fr_layout(n, &edges);
    let span = VIEW - 2.0 * MARGIN;
    let map = |p: (f64, f64)| (MARGIN + p.0 * span, MARGIN + p.1 * span);

    let mut s = String::new();
    let _ = write!(
        s,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{VIEW}\" height=\"{VIEW}\" \
         viewBox=\"0 0 {VIEW} {VIEW}\">"
    );
    // Pass 1: every other edge in the graph, faint — the ambient context.
    for &(a, b) in &edges {
        if a >= n || b >= n || highlight_edges.contains(&(a, b)) {
            continue;
        }
        let (x1, y1) = map(pos[a]);
        let (x2, y2) = map(pos[b]);
        let _ = write!(
            s,
            "<line x1=\"{x1:.1}\" y1=\"{y1:.1}\" x2=\"{x2:.1}\" y2=\"{y2:.1}\" \
             stroke=\"#e5e7eb\" stroke-width=\"1\"/>"
        );
    }
    // Pass 2: every other vertex, faint.
    for (i, &position) in pos.iter().enumerate().take(n) {
        if node_colors.get(i).copied().flatten().is_some() {
            continue;
        }
        let (x, y) = map(position);
        let _ = write!(
            s,
            "<circle cx=\"{x:.1}\" cy=\"{y:.1}\" r=\"4\" fill=\"#e5e7eb\" \
             stroke=\"#cbd5e1\" stroke-width=\"1\"/>"
        );
    }
    // Pass 3: the highlighted edges, bold, over the faint background.
    for &(a, b) in &highlight_edges {
        if a >= n || b >= n {
            continue;
        }
        let (x1, y1) = map(pos[a]);
        let (x2, y2) = map(pos[b]);
        let _ = write!(
            s,
            "<line x1=\"{x1:.1}\" y1=\"{y1:.1}\" x2=\"{x2:.1}\" y2=\"{y2:.1}\" \
             stroke=\"#2563eb\" stroke-width=\"2.5\"/>"
        );
    }
    // Pass 4: the highlighted vertices, on top of everything.
    for (i, color) in node_colors.iter().enumerate().take(n) {
        let Some(fill) = color else { continue };
        let (x, y) = map(pos[i]);
        let _ = write!(
            s,
            "<circle cx=\"{x:.1}\" cy=\"{y:.1}\" r=\"5.5\" fill=\"{fill}\" \
             stroke=\"#1e3a8a\" stroke-width=\"1\"/>"
        );
    }
    let _ = write!(s, "</svg>");
    Some(s)
}

/// Render a [`crate::py::graph::PyShortestPath`] highlighted within its
/// ambient graph: `(n, edges, directed)` are the *source* graph's — captured
/// at `Graph.shortest_path` construction time, since `ShortestPath` itself
/// holds no reference back to the graph it came from — drawn faint, with
/// `path`'s edges bold and vertices colored (source green, target red,
/// intermediate path vertices accented blue; a single-vertex path, `s == t`,
/// is just green). `None` if the ambient graph is empty, larger than
/// [`MAX_SVG_NODES`], or `path` is empty.
pub(crate) fn shortest_path_svg(
    n: usize,
    edges: Vec<(usize, usize)>,
    directed: bool,
    path: &[usize],
) -> Option<String> {
    if path.is_empty() {
        return None;
    }
    let mut colors: Vec<Option<&str>> = vec![None; n];
    for &v in path {
        if v < n {
            colors[v] = Some("#3b82f6");
        }
    }
    if let Some(&src) = path.first() {
        if src < n {
            colors[src] = Some("#16a34a");
        }
    }
    if path.len() > 1 {
        if let Some(&tgt) = path.last() {
            if tgt < n {
                colors[tgt] = Some("#dc2626");
            }
        }
    }
    let highlight: std::collections::HashSet<(usize, usize)> =
        path.windows(2).map(|w| (w[0], w[1])).collect();
    node_link_svg_highlighted(n, edges, directed, &colors, highlight)
}

/// Render a [`crate::py::graph::PyTraversal`]'s predecessor tree highlighted
/// within its ambient graph: `(n, edges, directed)` are the *source* graph's
/// (same capture-at-construction-time rationale as [`shortest_path_svg`]),
/// drawn faint, with the `pred` tree's edges bold and reached vertices
/// colored (source green, other reached vertices blue) — leaving unreached
/// vertices in the faint background, so the diagram shows both what the
/// traversal covered *and* what it didn't. `pred` is indexed by the
/// ambient graph's `VertexId`, per [`crate::py::graph::PyTraversal::pred`]'s
/// contract. `None` for an empty traversal or an ambient graph larger than
/// [`MAX_SVG_NODES`].
pub(crate) fn traversal_tree_svg(
    n: usize,
    edges: Vec<(usize, usize)>,
    directed: bool,
    order: &[usize],
    pred: &[Option<usize>],
) -> Option<String> {
    if order.is_empty() {
        return None;
    }
    let mut colors: Vec<Option<&str>> = vec![None; n];
    for (i, &v) in order.iter().enumerate() {
        if v < n {
            colors[v] = Some(if i == 0 { "#16a34a" } else { "#3b82f6" });
        }
    }
    let mut highlight = std::collections::HashSet::with_capacity(order.len());
    for &v in order {
        if let Some(Some(p)) = pred.get(v) {
            highlight.insert((*p, v));
        }
    }
    node_link_svg_highlighted(n, edges, directed, &colors, highlight)
}
