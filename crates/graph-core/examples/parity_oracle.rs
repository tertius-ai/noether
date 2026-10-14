// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Noether side of the `graph-core` cross-library parity harness.
//!
//! Reads a graph spec as JSON on stdin, computes the harness's
//! comparable invariants with `graph-core`, and writes them as JSON
//! on stdout. The Python harness under `validation/` drives
//! this binary and compares its output against NetworkX / igraph /
//! Sage / Magma on the same graph.
//!
//! The plain `examples/` binary uses hand-rolled minimal JSON so parity support
//! adds no runtime dependency to the kernel.
//!
//! # Input schema
//!
//! ```json
//! { "n": 8,
//!   "directed": false,
//!   "edges": [[0,1,1],[1,2,1]],   // [source, target, weight]
//!   "sources": [0],               // BFS/DFS roots to report
//!   "paths": [[0,4]],             // shortest-path (s,t) queries
//!   "subgraph_subset": [0,1,2],   // optional: induced subgraph cells
//!   "vertex_labels": [10,20,...], // optional: per-vertex int labels
//!   "edge_labels":   [1,2,...] }  // optional: per-edge int labels
//! ```
//!
//! When `subgraph_subset` is present, the oracle reports the same
//! comparable invariants on the induced subgraph (component count +
//! labels + BFS depth from `0` within the sub) plus, if labels are
//! attached, the carried-over vertex/edge label lists. When the input
//! is directed, the same invariants are also reported on the
//! transpose, alongside transpose-of-labels.
//!
//! # Output schema
//!
//! ```json
//! { "order": 8,
//!   "edge_count": 2,
//!   "component_count": 6,
//!   "component_labels": [0,0,0,1,2,3,4,5],
//!   "bfs": { "0": { "order": [...], "depth": [...] } },
//!   "dfs": { "0": { "order": [...] } },
//!   "paths": { "0-4": { "distance": 4, "path": [...] } },
//!   "subgraph": {                       // present iff subset given
//!       "order": 3, "edge_count": 1,
//!       "component_labels": [0,0,1],
//!       "vertex_map": [0,1,2],          // new-id i was old-id vmap[i]
//!       "vertex_labels": [10,20,30],    // present iff labels attached
//!       "edge_labels":   [7]
//!   },
//!   "transpose": {                      // present iff directed
//!       "order": 8, "edge_count": 2,
//!       "component_labels": [...],
//!       "vertex_labels": [...],         // = original (labels invariant
//!       "edge_labels":   [...]          //   under transpose)
//!   } }
//! ```
//!
//! A `null` path / depth entry means unreachable.

use graph_core::{
    bfs, connected_components, dfs, graph_from_edges, shortest_path, subgraph, transpose, Labeling,
    VertexId,
};
use std::io::Read;

fn main() {
    let mut input = String::new();
    if std::io::stdin().read_to_string(&mut input).is_err() {
        eprintln!("parity_oracle: failed to read stdin");
        std::process::exit(1);
    }
    match run(&input) {
        Ok(json) => println!("{json}"),
        Err(e) => {
            eprintln!("parity_oracle: {e}");
            std::process::exit(1);
        }
    }
}

/// Parse the request, run the queries, render the response JSON.
fn run(input: &str) -> Result<String, String> {
    let req = json::parse(input)?;

    let n = req.get_int("n")? as usize;
    let directed = req.get_bool("directed")?;
    let edges = req.get_edges("edges")?;
    let sources = req.get_int_list("sources").unwrap_or_default();
    let paths = req.get_pair_list("paths").unwrap_or_default();
    let subgraph_subset = req.get_int_list("subgraph_subset");
    let vertex_labels = req.get_i64_list("vertex_labels");
    let edge_labels = req.get_i64_list("edge_labels");

    let g = graph_from_edges(n, &edges, directed)
        .map_err(|e| format!("graph construction failed: {e}"))?;

    let mut out = String::from("{");
    out.push_str(&format!("\"order\":{}", g.order()));
    out.push_str(&format!(",\"edge_count\":{}", g.edge_count()));

    // Component invariants.
    let comps = connected_components(&g);
    out.push_str(&format!(",\"component_count\":{}", comps.count));
    out.push_str(",\"component_labels\":");
    out.push_str(&usize_array(&comps.label));

    // BFS / DFS from each requested source.
    out.push_str(",\"bfs\":{");
    for (i, &s) in sources.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let t = bfs(&g, s).map_err(|e| format!("bfs({s}): {e}"))?;
        out.push_str(&format!("\"{s}\":{{\"order\":"));
        out.push_str(&vertexid_array(&t.order));
        out.push_str(",\"depth\":");
        out.push_str(&opt_usize_array(&t.depths()));
        out.push('}');
    }
    out.push('}');

    out.push_str(",\"dfs\":{");
    for (i, &s) in sources.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let t = dfs(&g, s).map_err(|e| format!("dfs({s}): {e}"))?;
        out.push_str(&format!("\"{s}\":{{\"order\":"));
        out.push_str(&vertexid_array(&t.order));
        out.push('}');
    }
    out.push('}');

    // Shortest-path queries.
    out.push_str(",\"paths\":{");
    for (i, &(s, t)) in paths.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let sp = shortest_path(&g, s, t).map_err(|e| format!("shortest_path({s},{t}): {e}"))?;
        out.push_str(&format!("\"{s}-{t}\":"));
        match sp {
            None => out.push_str("null"),
            Some(p) => {
                out.push_str(&format!("{{\"distance\":{},\"path\":", p.distance));
                out.push_str(&vertexid_array(&p.path));
                out.push('}');
            }
        }
    }
    out.push('}');

    // Labeling: attach labels if both lists were supplied with matching
    // lengths. The kernel rejects mismatched sizes, which would taint
    // the response — make it an explicit error so the harness records
    // it as such rather than mis-parsing.
    let labeling: Option<Labeling<i64, i64>> = match (&vertex_labels, &edge_labels) {
        (Some(vl), Some(el)) => {
            Some(Labeling::new(&g, vl, el).map_err(|e| format!("labeling: {e}"))?)
        }
        _ => None,
    };

    // `subgraph` invariants — present iff a subset was requested.
    if let Some(subset) = subgraph_subset.as_ref() {
        let (sub, vmap) = subgraph(&g, subset).map_err(|e| format!("subgraph: {e}"))?;
        let sub_comps = connected_components(&sub);
        out.push_str(",\"subgraph\":{");
        out.push_str(&format!("\"order\":{}", sub.order()));
        out.push_str(&format!(",\"edge_count\":{}", sub.edge_count()));
        out.push_str(&format!(",\"component_count\":{}", sub_comps.count));
        out.push_str(",\"component_labels\":");
        out.push_str(&usize_array(&sub_comps.label));
        out.push_str(",\"vertex_map\":");
        out.push_str(&vertexid_array(&vmap));
        // Carry labels through the kernel's subgraph route — the
        // parity check is the kernel's "labels preserved under
        // subgraph" guarantee, not a reimplementation in the harness.
        if let Some(lab) = labeling.as_ref() {
            let sublab = lab
                .subgraph(&g, &sub, &vmap)
                .map_err(|e| format!("subgraph labels: {e}"))?;
            out.push_str(",\"vertex_labels\":");
            out.push_str(&i64_array(sublab.vertex_labels()));
            out.push_str(",\"edge_labels\":");
            out.push_str(&i64_array(sublab.edge_labels()));
        }
        out.push('}');
    }

    // `transpose` invariants — present iff the input is directed
    // (undirected transpose is the identity, so it would add no
    // signal to the parity sweep). For Labeling, transposed labels =
    // original labels. The harness exposes them so each oracle can
    // verify its own transpose against the same invariant.
    if directed {
        let gt = transpose(&g);
        let t_comps = connected_components(&gt);
        out.push_str(",\"transpose\":{");
        out.push_str(&format!("\"order\":{}", gt.order()));
        out.push_str(&format!(",\"edge_count\":{}", gt.edge_count()));
        out.push_str(&format!(",\"component_count\":{}", t_comps.count));
        out.push_str(",\"component_labels\":");
        out.push_str(&usize_array(&t_comps.label));
        if let Some(lab) = labeling.as_ref() {
            let labt = lab.transpose();
            out.push_str(",\"vertex_labels\":");
            out.push_str(&i64_array(labt.vertex_labels()));
            out.push_str(",\"edge_labels\":");
            out.push_str(&i64_array(labt.edge_labels()));
        }
        out.push('}');
    }

    out.push('}');
    Ok(out)
}

fn usize_array(v: &[usize]) -> String {
    let body: Vec<String> = v.iter().map(usize::to_string).collect();
    format!("[{}]", body.join(","))
}

fn vertexid_array(v: &[VertexId]) -> String {
    let body: Vec<String> = v.iter().map(VertexId::to_string).collect();
    format!("[{}]", body.join(","))
}

fn opt_usize_array(v: &[Option<usize>]) -> String {
    let body: Vec<String> = v
        .iter()
        .map(|x| x.map_or_else(|| "null".to_string(), |d| d.to_string()))
        .collect();
    format!("[{}]", body.join(","))
}

fn i64_array(v: &[i64]) -> String {
    let body: Vec<String> = v.iter().map(i64::to_string).collect();
    format!("[{}]", body.join(","))
}

/// A deliberately tiny, dependency-free JSON reader — just enough to
/// parse the oracle's request schema. Not a general JSON library.
mod json {
    use graph_core::VertexId;

    /// A parsed JSON value.
    ///
    /// The parser is a general (if tiny) JSON reader, so it
    /// represents every JSON shape; the request schema happens never
    /// to read a string *value* (only string object keys, which are
    /// stored separately), so `Str`'s payload is unread — kept so the
    /// parser accepts valid JSON rather than rejecting it.
    #[derive(Debug, Clone)]
    pub enum Value {
        Null,
        Bool(bool),
        Num(f64),
        Str(#[allow(dead_code)] String),
        Arr(Vec<Value>),
        Obj(Vec<(String, Value)>),
    }

    /// Parse a complete JSON document.
    pub fn parse(s: &str) -> Result<Value, String> {
        let mut p = Parser {
            bytes: s.as_bytes(),
            pos: 0,
        };
        p.skip_ws();
        let v = p.value()?;
        p.skip_ws();
        if p.pos != p.bytes.len() {
            return Err("trailing characters after JSON document".into());
        }
        Ok(v)
    }

    impl Value {
        /// Integer field of an object, by key.
        pub fn get_int(&self, key: &str) -> Result<i64, String> {
            match self.field(key)? {
                Value::Num(n) => Ok(*n as i64),
                _ => Err(format!("field `{key}` is not a number")),
            }
        }

        /// Boolean field of an object, by key.
        pub fn get_bool(&self, key: &str) -> Result<bool, String> {
            match self.field(key)? {
                Value::Bool(b) => Ok(*b),
                _ => Err(format!("field `{key}` is not a bool")),
            }
        }

        /// `(source, target, weight)` triples from an `edges` array.
        ///
        /// Vertex endpoints are read into [`VertexId`]. Parity fixtures keep
        /// every id within the `u32` range.
        pub fn get_edges(&self, key: &str) -> Result<Vec<(VertexId, VertexId, i64)>, String> {
            let arr = match self.field(key)? {
                Value::Arr(a) => a,
                _ => return Err(format!("field `{key}` is not an array")),
            };
            let mut out = Vec::with_capacity(arr.len());
            for e in arr {
                let triple = match e {
                    Value::Arr(t) if t.len() == 3 => t,
                    _ => return Err("each edge must be [src,tgt,weight]".into()),
                };
                out.push((
                    as_num(&triple[0])? as VertexId,
                    as_num(&triple[1])? as VertexId,
                    as_num(&triple[2])? as i64,
                ));
            }
            Ok(out)
        }

        /// A flat list of vertex ids from an array field; absent → `None`.
        pub fn get_int_list(&self, key: &str) -> Option<Vec<VertexId>> {
            match self.field(key).ok()? {
                Value::Arr(a) => {
                    let mut out = Vec::with_capacity(a.len());
                    for v in a {
                        out.push(as_num(v).ok()? as VertexId);
                    }
                    Some(out)
                }
                _ => None,
            }
        }

        /// A flat list of signed integers from an array field; absent → `None`.
        ///
        /// Used for the optional `vertex_labels` / `edge_labels` request
        /// fields where labels are arbitrary `i64` tags.
        pub fn get_i64_list(&self, key: &str) -> Option<Vec<i64>> {
            match self.field(key).ok()? {
                Value::Arr(a) => {
                    let mut out = Vec::with_capacity(a.len());
                    for v in a {
                        out.push(as_num(v).ok()? as i64);
                    }
                    Some(out)
                }
                _ => None,
            }
        }

        /// A list of `(s, t)` vertex pairs from an array field; absent → `None`.
        pub fn get_pair_list(&self, key: &str) -> Option<Vec<(VertexId, VertexId)>> {
            match self.field(key).ok()? {
                Value::Arr(a) => {
                    let mut out = Vec::with_capacity(a.len());
                    for v in a {
                        match v {
                            Value::Arr(p) if p.len() == 2 => out.push((
                                as_num(&p[0]).ok()? as VertexId,
                                as_num(&p[1]).ok()? as VertexId,
                            )),
                            _ => return None,
                        }
                    }
                    Some(out)
                }
                _ => None,
            }
        }

        fn field(&self, key: &str) -> Result<&Value, String> {
            match self {
                Value::Obj(pairs) => pairs
                    .iter()
                    .find(|(k, _)| k == key)
                    .map(|(_, v)| v)
                    .ok_or_else(|| format!("missing field `{key}`")),
                _ => Err("expected a JSON object".into()),
            }
        }
    }

    fn as_num(v: &Value) -> Result<f64, String> {
        match v {
            Value::Num(n) => Ok(*n),
            _ => Err("expected a number".into()),
        }
    }

    struct Parser<'a> {
        bytes: &'a [u8],
        pos: usize,
    }

    impl Parser<'_> {
        fn skip_ws(&mut self) {
            while self.pos < self.bytes.len()
                && matches!(self.bytes[self.pos], b' ' | b'\t' | b'\n' | b'\r')
            {
                self.pos += 1;
            }
        }

        fn value(&mut self) -> Result<Value, String> {
            self.skip_ws();
            match self.bytes.get(self.pos) {
                Some(b'{') => self.object(),
                Some(b'[') => self.array(),
                Some(b'"') => Ok(Value::Str(self.string()?)),
                Some(b't') | Some(b'f') => self.boolean(),
                Some(b'n') => self.null(),
                Some(c) if *c == b'-' || c.is_ascii_digit() => self.number(),
                _ => Err("unexpected token".into()),
            }
        }

        fn object(&mut self) -> Result<Value, String> {
            self.pos += 1; // consume '{'
            let mut pairs = Vec::new();
            self.skip_ws();
            if self.bytes.get(self.pos) == Some(&b'}') {
                self.pos += 1;
                return Ok(Value::Obj(pairs));
            }
            loop {
                self.skip_ws();
                let key = self.string()?;
                self.skip_ws();
                if self.bytes.get(self.pos) != Some(&b':') {
                    return Err("expected `:` in object".into());
                }
                self.pos += 1;
                let val = self.value()?;
                pairs.push((key, val));
                self.skip_ws();
                match self.bytes.get(self.pos) {
                    Some(b',') => self.pos += 1,
                    Some(b'}') => {
                        self.pos += 1;
                        return Ok(Value::Obj(pairs));
                    }
                    _ => return Err("expected `,` or `}` in object".into()),
                }
            }
        }

        fn array(&mut self) -> Result<Value, String> {
            self.pos += 1; // consume '['
            let mut items = Vec::new();
            self.skip_ws();
            if self.bytes.get(self.pos) == Some(&b']') {
                self.pos += 1;
                return Ok(Value::Arr(items));
            }
            loop {
                items.push(self.value()?);
                self.skip_ws();
                match self.bytes.get(self.pos) {
                    Some(b',') => self.pos += 1,
                    Some(b']') => {
                        self.pos += 1;
                        return Ok(Value::Arr(items));
                    }
                    _ => return Err("expected `,` or `]` in array".into()),
                }
            }
        }

        fn string(&mut self) -> Result<String, String> {
            if self.bytes.get(self.pos) != Some(&b'"') {
                return Err("expected a string".into());
            }
            self.pos += 1;
            let start = self.pos;
            while self.pos < self.bytes.len() && self.bytes[self.pos] != b'"' {
                // The request schema has no escaped strings; reject
                // them rather than mis-parse.
                if self.bytes[self.pos] == b'\\' {
                    return Err("escaped strings are not supported".into());
                }
                self.pos += 1;
            }
            if self.pos >= self.bytes.len() {
                return Err("unterminated string".into());
            }
            let s = std::str::from_utf8(&self.bytes[start..self.pos])
                .map_err(|_| "invalid UTF-8 in string".to_string())?
                .to_string();
            self.pos += 1; // consume closing '"'
            Ok(s)
        }

        fn number(&mut self) -> Result<Value, String> {
            let start = self.pos;
            if self.bytes.get(self.pos) == Some(&b'-') {
                self.pos += 1;
            }
            while self.pos < self.bytes.len()
                && (self.bytes[self.pos].is_ascii_digit()
                    || matches!(self.bytes[self.pos], b'.' | b'e' | b'E' | b'+' | b'-'))
            {
                self.pos += 1;
            }
            let text = std::str::from_utf8(&self.bytes[start..self.pos])
                .map_err(|_| "invalid number".to_string())?;
            text.parse::<f64>()
                .map(Value::Num)
                .map_err(|_| format!("invalid number `{text}`"))
        }

        fn boolean(&mut self) -> Result<Value, String> {
            if self.bytes[self.pos..].starts_with(b"true") {
                self.pos += 4;
                Ok(Value::Bool(true))
            } else if self.bytes[self.pos..].starts_with(b"false") {
                self.pos += 5;
                Ok(Value::Bool(false))
            } else {
                Err("invalid boolean literal".into())
            }
        }

        fn null(&mut self) -> Result<Value, String> {
            if self.bytes[self.pos..].starts_with(b"null") {
                self.pos += 4;
                Ok(Value::Null)
            } else {
                Err("invalid null literal".into())
            }
        }
    }
}
