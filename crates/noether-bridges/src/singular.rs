// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Arm's-length `Singular` subprocess backend (commutative algebra / AG).
//!
//! [Singular](https://www.singular.uni-kl.de) is a GPL system specialising in
//! polynomial systems, free resolutions, and singularity theory. It is the
//! free-resolution engine, works over every field including `GF(q)`, and
//! returns full differential matrices. This module is the single
//! audited spawn point for it: it probes availability and evaluates a
//! caller-built Singular script, returning stdout. Links **no** GPL —
//! `Singular` is the user's binary, spawned arm's-length — so it is
//! conveyed-clean and mirrors the `macaulay2`/`msolve` subprocess tiers.
//!
//! The Singular script and output grammar are pure text with no GPL content.
//! Noether's Rust facade owns operation-specific templating, parsing, and exact
//! verification; this module only evaluates the script and returns raw stdout.

#![cfg(feature = "singular-subprocess")]

use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use crate::{isolate_native_libraries, BridgeError};

const SINGULAR_BIN: &str = "Singular";

/// Is a `Singular` binary reachable on `PATH`? Cheap probe (`Singular --version`).
///
/// The engine's dispatch calls this to decide whether the `singular-subprocess`
/// backend is live; on `false` the caller falls back to the native Schreyer path
/// (small inputs) or surfaces the typed unavailable.
pub fn singular_available() -> bool {
    match Command::new(SINGULAR_BIN)
        .arg("--version")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
    {
        Ok(_) => true,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => false,
        Err(_) => false,
    }
}

/// Evaluate a Singular `script` and return its stdout.
///
/// The script is written to a private temp `.sing` file and run as
/// `Singular -q <file>` (`-q` suppresses the banner / startup chatter). The
/// caller builds the script (the `ring` + `ideal` + op + a `print`/`string` of
/// the answer, ending in `quit;`) and parses the returned stdout. stdin is
/// closed (so a script missing its `quit;` exits on EOF rather than hanging);
/// stdout and stderr go to files, not pipes, so a large resolution can never
/// deadlock on a full pipe buffer.
///
/// `timeout_secs`, when `Some`, kills Singular after that many seconds — the
/// guard for hard inputs.
///
/// Returns [`BridgeError::Unavailable`] if `Singular` is not on `PATH` (so the
/// caller can fall back to native), [`BridgeError::NonZeroExit`] (carrying
/// stderr) if Singular exits non-zero or times out, and [`BridgeError::Parse`]
/// if its stdout cannot be read.
pub fn singular_eval(script: &str, timeout_secs: Option<f64>) -> Result<String, BridgeError> {
    let dir = tempfile::Builder::new()
        .prefix("noether-singular-")
        .tempdir()
        .map_err(BridgeError::Spawn)?;
    let script_path = dir.path().join("script.sing");
    let out_path = dir.path().join("out.txt");
    let err_path = dir.path().join("err.txt");
    std::fs::write(&script_path, script).map_err(BridgeError::Spawn)?;
    let out_file = std::fs::File::create(&out_path).map_err(BridgeError::Spawn)?;
    let err_file = std::fs::File::create(&err_path).map_err(BridgeError::Spawn)?;

    let mut command = Command::new(SINGULAR_BIN);
    command
        .arg("-q")
        .arg(&script_path)
        // Closed stdin: a script without a trailing `quit;` exits on EOF instead
        // of dropping into the interactive interpreter and hanging.
        .stdin(Stdio::null())
        .stdout(Stdio::from(out_file))
        .stderr(Stdio::from(err_file));
    // Do not let Noether's private FLINT/GMP search path alter Singular's ABI
    // stack.
    isolate_native_libraries(&mut command);
    let mut child = command.spawn().map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            BridgeError::Unavailable(format!("`{SINGULAR_BIN}` not found on PATH"))
        } else {
            BridgeError::Spawn(e)
        }
    })?;

    let status = match timeout_secs {
        None => child.wait().map_err(BridgeError::Spawn)?,
        Some(secs) => {
            let budget = Duration::try_from_secs_f64(secs).unwrap_or(Duration::ZERO);
            let deadline = Instant::now() + budget;
            loop {
                if let Some(s) = child.try_wait().map_err(BridgeError::Spawn)? {
                    break s;
                }
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(BridgeError::NonZeroExit {
                        code: None,
                        stderr: format!("Singular timed out after {secs}s"),
                    });
                }
                std::thread::sleep(Duration::from_millis(25));
            }
        }
    };

    if !status.success() {
        let stderr = std::fs::read_to_string(&err_path).unwrap_or_default();
        return Err(BridgeError::NonZeroExit {
            code: status.code(),
            stderr,
        });
    }

    std::fs::read_to_string(&out_path)
        .map_err(|e| BridgeError::Parse(format!("reading Singular stdout `{out_path:?}`: {e}")))
    // `dir` (and the temp files) are removed on drop here.
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Singular evaluates a trivial script and prints to stdout. Skips when absent.
    #[test]
    fn singular_eval_smoke() {
        if !singular_available() {
            eprintln!("skipping: `Singular` not on PATH");
            return;
        }
        let out = singular_eval(
            "ring r = 0,(x,y),dp; \"dim=\", dim(std(ideal(x))); quit;",
            Some(60.0),
        )
        .expect("Singular should evaluate a trivial script");
        // dim of V(x) in A^2 is 1.
        assert!(
            out.contains("dim= 1"),
            "unexpected Singular output: {out:?}"
        );
    }

    /// The minimal free resolution Betti numbers of the twisted cubic ideal
    /// `(xz-y^2, xw-yz, yw-z^2)` ⊂ Q[x,y,z,w] — the diagram `1; -,3; -,2`.
    /// Smoke-tests the exact `betti(mres(I,0))` path + intmat print format the
    /// Python backend parses.
    #[test]
    fn singular_twisted_cubic_betti() {
        if !singular_available() {
            eprintln!("skipping: `Singular` not on PATH");
            return;
        }
        let script = "ring r = 0,(x,y,z,w),dp; \
             ideal I = xz-y2, xw-yz, yw-z2; \
             resolution F = mres(I,0); intmat B = betti(F); \
             \"rows=\", nrows(B); \"cols=\", ncols(B); \
             int i,j; for(i=1;i<=nrows(B);i++){for(j=1;j<=ncols(B);j++){ \
               string(\"b \" + string(i) + \" \" + string(j) + \" \" + string(B[i,j])); }} quit;";
        let out = singular_eval(script, Some(60.0)).expect("Singular resolution should run");
        assert!(
            out.contains("cols= 3"),
            "expected pd 2 (3 columns), got: {out:?}"
        );
        assert!(out.contains("b 1 1 1"), "beta_00 should be 1, got: {out:?}");
        assert!(out.contains("b 2 2 3"), "beta_12 should be 3, got: {out:?}");
        assert!(out.contains("b 2 3 2"), "beta_23 should be 2, got: {out:?}");
    }

    #[test]
    fn unavailable_is_typed_not_panic() {
        let _ = singular_available();
    }
}
