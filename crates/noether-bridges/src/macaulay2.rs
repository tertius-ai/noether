// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Arm's-length `M2` (Macaulay2) subprocess backend (commutative algebra).
//!
//! [Macaulay2](https://macaulay2.com) is the gold-standard GPL commutative-
//! algebra system (resolutions, Betti tables, Hilbert series, dimension/degree,
//! primary decomposition, associated/minimal primes, radical, saturation, …).
//! This module is the single audited spawn point for it: it probes availability
//! and evaluates a caller-built M2 script, returning stdout. Links **no** GPL —
//! `M2` is the user's binary, spawned arm's-length via `M2 --script` — so it is
//! conveyed-clean and follows the crate's common subprocess contract.
//!
//! The M2 script and output grammar are pure text with no GPL content.
//! Noether's Rust facade owns operation-specific templating, parsing, and exact
//! verification; this module only evaluates the script and returns raw stdout.

#![cfg(feature = "macaulay2-subprocess")]

use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use crate::{isolate_native_libraries, BridgeError};

const M2_BIN: &str = "M2";

/// Is an `M2` binary reachable on `PATH`? Cheap probe (`M2 --version`).
///
/// The engine's dispatch calls this to decide whether the
/// `macaulay2-subprocess` backend is live; on `false` the caller surfaces the
/// typed unavailable (the wrapped ops have no native path yet).
pub fn m2_available() -> bool {
    match Command::new(M2_BIN)
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
    {
        Ok(_) => true,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => false,
        Err(_) => false,
    }
}

/// Evaluate an M2 `script` and return its stdout.
///
/// The script is written to a private temp `.m2` file and run as
/// `M2 --script <file>` (which implies `--stop --no-debug --silent -q`, so the
/// process exits after the script and emits no banner). The caller builds the
/// script (ring + ideal + op + a `print`/`<<` of the answer) and parses the
/// returned stdout. stdout and stderr are redirected to files (not pipes), so a
/// large resolution / long run can never deadlock on a full pipe buffer.
///
/// `timeout_secs`, when `Some`, kills M2 after that many seconds.
///
/// Returns [`BridgeError::Unavailable`] if `M2` is not on `PATH`, and
/// [`BridgeError::NonZeroExit`] (carrying stderr) if the script errors or times
/// out — which is how an M2-version API drift surfaces to the caller rather
/// than as a silent empty result.
pub fn m2_eval(script: &str, timeout_secs: Option<f64>) -> Result<String, BridgeError> {
    let dir = tempfile::Builder::new()
        .prefix("noether-m2-")
        .tempdir()
        .map_err(BridgeError::Spawn)?;
    let script_path = dir.path().join("script.m2");
    let out_path = dir.path().join("out.txt");
    let err_path = dir.path().join("err.txt");
    std::fs::write(&script_path, script).map_err(BridgeError::Spawn)?;
    let out_file = std::fs::File::create(&out_path).map_err(BridgeError::Spawn)?;
    let err_file = std::fs::File::create(&err_path).map_err(BridgeError::Spawn)?;

    let mut command = Command::new(M2_BIN);
    command
        .arg("--script")
        .arg(&script_path)
        .stdout(Stdio::from(out_file))
        .stderr(Stdio::from(err_file));
    // Do not let Noether's private FLINT/GMP search path alter M2's ABI stack.
    isolate_native_libraries(&mut command);
    let mut child = command.spawn().map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            BridgeError::Unavailable(format!("`{M2_BIN}` not found on PATH"))
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
                        stderr: format!("M2 timed out after {secs}s"),
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
        .map_err(|e| BridgeError::Parse(format!("reading M2 stdout `{out_path:?}`: {e}")))
    // `dir` (and the temp files) are removed on drop here.
}

#[cfg(test)]
mod tests {
    use super::*;

    /// M2 evaluates a trivial script and prints to stdout. Skips when absent.
    #[test]
    fn m2_eval_smoke() {
        if !m2_available() {
            eprintln!("skipping: `M2` not on PATH");
            return;
        }
        let out = m2_eval(
            "R = QQ[x,y]; I = ideal(x^2 - y); << dim(R/I) << endl;",
            Some(60.0),
        )
        .expect("M2 should evaluate a trivial script");
        // dim of a plane curve in 2 vars is 1.
        assert!(out.contains('1'), "unexpected M2 output: {out:?}");
    }

    #[test]
    fn m2_script_error_is_nonzero_exit() {
        if !m2_available() {
            eprintln!("skipping: `M2` not on PATH");
            return;
        }
        // A deliberate M2 error must surface as NonZeroExit with stderr, not a
        // silent success.
        let r = m2_eval("error \"boom\";", Some(60.0));
        assert!(
            matches!(r, Err(BridgeError::NonZeroExit { .. })),
            "got {r:?}"
        );
    }

    #[test]
    fn unavailable_is_typed_not_panic() {
        let _ = m2_available();
    }
}
