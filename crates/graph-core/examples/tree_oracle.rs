// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! `tree_oracle` — the Tertius side of the `tree-core` cross-CAS parity
//! harness.
//!
//! Reads a tree spec as JSON on stdin, builds a `graph_core::Tree`, and
//! writes the harness's comparable invariants as JSON on stdout. The
//! Python harness under `validation/parity/combinatorics/`
//! (`test_tree_core_parity.py`) drives this binary and compares its
//! output against **Sage** (`Graph.is_tree`/`is_forest`/`diameter`/
//! `breadth_first_search`/`shortest_path_lengths`) and **NetworkX**.
//!
//! Hand-rolled minimal JSON, no serde — preserving the crate's
//! zero-dependency pure-Rust kernel identity (the `parity_oracle`
//! pattern).
//!
//! # Input schema
//!
//! ```json
//! { "n": 8,
//! "edges": [[0,1],[1,2],...], // n-1 undirected pairs (a tree)
//! "root": 0, // root vertex (default 0)
//! "lca_pairs": [[3,5],[2,7]], // (u,v) LCA queries
//! "anc_pairs": [[0,3]], // (a,d) is_ancestor queries
//! "subtree_vs": [1,2], // subtree_size + subtree queries
//! "kth": [[5,1],[7,2]] } // (v,k) kth_ancestor queries
//! ```
//!
//! # Output schema
//!
//! ```json
//! { "is_tree": true, "is_forest": true,
//! "order": 8, "root": 0,
//! "depth": [0,1,2,...], // per vertex
//! "subtree_size": [8,3,1,...], // per vertex
//! "descendants": { "1": [1,3,4] }, // for each subtree_vs vertex
//! "lca": { "3-5": 1 }, // per lca pair
//! "is_ancestor": { "0-3": true }, // per anc pair
//! "kth_ancestor": { "5-1": 2 }, // per kth query (-1 = None)
//! "diameter": 4, "diameter_path": [...],
//! "centroid": [3], // size-balance centroid(s)
//! "cd_depth": 3, // centroid-decomposition depth
//! "hld_max_light": 2 } // max light edges root→leaf
//! ```

use graph_core::{is_forest, is_tree, tree_from_edges, Graph, VertexId};
use std::io::Read;

fn main() {
    let mut input = String::new();
    std::io::stdin()
        .read_to_string(&mut input)
        .expect("read stdin");
    let spec = parse_spec(&input);

    // Build the graph form too, for is_tree/is_forest on the arbitrary
    // graph (the predicate surface).
    let weighted: Vec<(VertexId, VertexId, i64)> =
        spec.edges.iter().map(|&(u, v)| (u, v, 1)).collect();
    let graph = Graph::from_edges(spec.n, &weighted, false).expect("valid endpoints");
    let tree_is = is_tree(&graph);
    let forest_is = is_forest(&graph);

    let tree = tree_from_edges(spec.n, &spec.edges, Some(spec.root)).expect("spec is a tree");
    let n = spec.n;

    let depth: Vec<usize> = (0..n).map(|v| tree.depth(v as VertexId)).collect();
    let subtree_size: Vec<usize> = (0..n).map(|v| tree.subtree_size(v as VertexId)).collect();

    let mut out = String::new();
    out.push('{');
    push_kv_bool(&mut out, "is_tree", tree_is);
    out.push(',');
    push_kv_bool(&mut out, "is_forest", forest_is);
    out.push(',');
    push_kv_usize(&mut out, "order", n);
    out.push(',');
    push_kv_usize(&mut out, "root", spec.root as usize);
    out.push(',');
    push_kv_usize_list(&mut out, "depth", &depth);
    out.push(',');
    push_kv_usize_list(&mut out, "subtree_size", &subtree_size);

    // descendants (sorted ascending) for each requested subtree vertex.
    out.push_str(",\"descendants\":{");
    for (i, &v) in spec.subtree_vs.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let mut sub: Vec<VertexId> = tree.subtree(v).to_vec();
        sub.sort_unstable();
        out.push('"');
        out.push_str(&v.to_string());
        out.push_str("\":");
        push_u32_list(&mut out, &sub);
    }
    out.push('}');

    // lca per pair.
    out.push_str(",\"lca\":{");
    for (i, &(u, v)) in spec.lca_pairs.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push('"');
        out.push_str(&format!("{u}-{v}"));
        out.push_str("\":");
        out.push_str(&tree.lca(u, v).to_string());
    }
    out.push('}');

    // is_ancestor per pair.
    out.push_str(",\"is_ancestor\":{");
    for (i, &(a, d)) in spec.anc_pairs.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push('"');
        out.push_str(&format!("{a}-{d}"));
        out.push_str("\":");
        out.push_str(if tree.is_ancestor(a, d) {
            "true"
        } else {
            "false"
        });
    }
    out.push('}');

    // kth_ancestor per query (-1 for None).
    out.push_str(",\"kth_ancestor\":{");
    for (i, &(v, k)) in spec.kth.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push('"');
        out.push_str(&format!("{v}-{k}"));
        out.push_str("\":");
        match tree.kth_ancestor(v, k) {
            Some(a) => out.push_str(&a.to_string()),
            None => out.push_str("-1"),
        }
    }
    out.push('}');

    // diameter + path.
    let (dlen, dpath) = tree.diameter();
    out.push(',');
    push_kv_usize(&mut out, "diameter", dlen);
    out.push_str(",\"diameter_path\":");
    push_u32_list(&mut out, &dpath);

    // centroid (sorted).
    let mut cents: Vec<VertexId> = tree.centroid().as_slice().to_vec();
    cents.sort_unstable();
    out.push_str(",\"centroid\":");
    push_u32_list(&mut out, &cents);

    // centroid-decomposition depth.
    let cd = tree.centroid_decomposition();
    out.push(',');
    push_kv_usize(&mut out, "cd_depth", cd.depth());

    // HLD max light edges over all vertices (invariant check ≤ ceil log2 n).
    let hld = tree.heavy_light_decomposition();
    let max_light = (0..n)
        .map(|v| hld.light_edges_to_root(v as VertexId))
        .max()
        .unwrap_or(0);
    out.push(',');
    push_kv_usize(&mut out, "hld_max_light", max_light);

    out.push('}');
    println!("{out}");
}

// ---------- minimal JSON helpers (hand-rolled, no serde) ----------

struct Spec {
    n: usize,
    edges: Vec<(VertexId, VertexId)>,
    root: VertexId,
    lca_pairs: Vec<(VertexId, VertexId)>,
    anc_pairs: Vec<(VertexId, VertexId)>,
    subtree_vs: Vec<VertexId>,
    kth: Vec<(VertexId, usize)>,
}

/// Parse the input spec. The harness emits compact, well-formed JSON, so a
/// permissive number-and-bracket scanner suffices (no general JSON parser).
fn parse_spec(s: &str) -> Spec {
    Spec {
        n: scan_scalar(s, "\"n\"").unwrap_or(0) as usize,
        edges: scan_pairs(s, "\"edges\"")
            .into_iter()
            .map(|(a, b)| (a as VertexId, b as VertexId))
            .collect(),
        root: scan_scalar(s, "\"root\"").unwrap_or(0) as VertexId,
        lca_pairs: scan_pairs(s, "\"lca_pairs\"")
            .into_iter()
            .map(|(a, b)| (a as VertexId, b as VertexId))
            .collect(),
        anc_pairs: scan_pairs(s, "\"anc_pairs\"")
            .into_iter()
            .map(|(a, b)| (a as VertexId, b as VertexId))
            .collect(),
        subtree_vs: scan_flat(s, "\"subtree_vs\"")
            .into_iter()
            .map(|x| x as VertexId)
            .collect(),
        kth: scan_pairs(s, "\"kth\"")
            .into_iter()
            .map(|(a, b)| (a as VertexId, b as usize))
            .collect(),
    }
}

/// Scan a scalar integer value following `key` (e.g. `"n": 8`).
fn scan_scalar(s: &str, key: &str) -> Option<i64> {
    let pos = s.find(key)?;
    let after = &s[pos + key.len()..];
    let colon = after.find(':')?;
    let rest = &after[colon + 1..];
    let mut num = String::new();
    for c in rest.chars() {
        if c.is_ascii_digit() || c == '-' {
            num.push(c);
        } else if !num.is_empty() {
            break;
        } else if c == ' ' || c == '\t' {
            continue;
        } else {
            break;
        }
    }
    num.parse().ok()
}

/// Scan a flat integer array following `key` (e.g. `"subtree_vs": [1,2]`).
fn scan_flat(s: &str, key: &str) -> Vec<i64> {
    let Some(pos) = s.find(key) else {
        return Vec::new();
    };
    let after = &s[pos + key.len()..];
    let Some(open) = after.find('[') else {
        return Vec::new();
    };
    let Some(close) = after[open..].find(']') else {
        return Vec::new();
    };
    parse_ints(&after[open + 1..open + close])
}

/// Scan an array of integer pairs following `key`
/// (e.g. `"edges": [[0,1],[1,2]]`).
fn scan_pairs(s: &str, key: &str) -> Vec<(i64, i64)> {
    let Some(pos) = s.find(key) else {
        return Vec::new();
    };
    let after = &s[pos + key.len()..];
    let Some(open) = after.find('[') else {
        return Vec::new();
    };
    // Find the matching outer ']'.
    let bytes = after.as_bytes();
    let mut depth = 0i32;
    let mut end = open;
    for (i, &b) in bytes.iter().enumerate().skip(open) {
        match b {
            b'[' => depth += 1,
            b']' => {
                depth -= 1;
                if depth == 0 {
                    end = i;
                    break;
                }
            }
            _ => {}
        }
    }
    let inner = &after[open + 1..end];
    let mut out = Vec::new();
    let mut i = 0;
    let ib = inner.as_bytes();
    while i < ib.len() {
        if ib[i] == b'[' {
            let mut j = i + 1;
            while j < ib.len() && ib[j] != b']' {
                j += 1;
            }
            let nums = parse_ints(&inner[i + 1..j]);
            if nums.len() >= 2 {
                out.push((nums[0], nums[1]));
            }
            i = j + 1;
        } else {
            i += 1;
        }
    }
    out
}

/// Parse a comma-separated list of integers from a fragment.
fn parse_ints(frag: &str) -> Vec<i64> {
    frag.split(',')
        .filter_map(|t| t.trim().parse::<i64>().ok())
        .collect()
}

fn push_kv_bool(out: &mut String, key: &str, v: bool) {
    out.push('"');
    out.push_str(key);
    out.push_str("\":");
    out.push_str(if v { "true" } else { "false" });
}

fn push_kv_usize(out: &mut String, key: &str, v: usize) {
    out.push('"');
    out.push_str(key);
    out.push_str("\":");
    out.push_str(&v.to_string());
}

fn push_kv_usize_list(out: &mut String, key: &str, v: &[usize]) {
    out.push('"');
    out.push_str(key);
    out.push_str("\":[");
    for (i, x) in v.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&x.to_string());
    }
    out.push(']');
}

fn push_u32_list(out: &mut String, v: &[VertexId]) {
    out.push('[');
    for (i, x) in v.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&x.to_string());
    }
    out.push(']');
}
