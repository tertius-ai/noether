// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Arm's-length `msolve` subprocess backend (commutative algebra).
//!
//! [msolve](https://github.com/algebraic-solving/msolve) is a fast GPL F4 /
//! real-solving engine. This module is the single audited spawn point for it:
//! it probes availability and runs the binary on a caller-built input body,
//! returning the raw output. Links **no** GPL — `msolve` is the user's binary,
//! spawned arm's-length — so it is conveyed-clean and follows the crate's
//! common subprocess contract.
//!
//! The CA input/output grammar is pure text with no GPL content. Noether's Rust
//! facade owns that operation-specific translation and verifies any imported
//! candidate; this module only spawns [`run_msolve`] and returns raw transport.

#![cfg(feature = "msolve-subprocess")]

use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use crate::{isolate_native_libraries, BridgeError};

const MSOLVE_BIN: &str = "msolve";

/// Is an `msolve` binary reachable on `PATH`? Cheap probe (`msolve -h`).
///
/// The engine's dispatch calls this to decide whether the `msolve-subprocess`
/// backend is live; on `false` the caller falls back to the native F4 path
/// (Gröbner) or surfaces the typed unavailable (zero-dim solve).
pub fn msolve_available() -> bool {
    // `-h` prints usage; any spawn that is not "binary missing" means present
    // (msolve's `-h` may exit non-zero on some builds, so we key on spawn, not
    // exit status).
    match Command::new(MSOLVE_BIN)
        .arg("-h")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
    {
        Ok(_) => true,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => false,
        Err(_) => false,
    }
}

/// Run `msolve` on `input_body` with the given mode arguments and return the
/// contents of msolve's output file.
///
/// `mode_args` selects the computation — e.g. `["-g", "2"]` for a reduced
/// Gröbner basis (degrevlex), or `["-P", "1"]` for the rational univariate
/// representation of a zero-dimensional system. The body is written to a
/// private temp `in.ms`; msolve is invoked as
/// `msolve <mode_args> -f <in.ms> -o <out.ms>`; `out.ms` is returned verbatim
/// for the caller to parse.
///
/// `timeout_secs`, when `Some`, kills msolve after that many seconds — the
/// guard for hard inputs that would otherwise run unbounded (msolve's stderr is
/// redirected to a file, not a pipe, so a long run can never deadlock on a full
/// pipe buffer).
///
/// Returns [`BridgeError::Unavailable`] if `msolve` is not on `PATH` (so the
/// caller can fall through to native), [`BridgeError::NonZeroExit`] if msolve
/// fails or times out, and [`BridgeError::Parse`] if its output file cannot be
/// read.
pub fn run_msolve(
    input_body: &str,
    mode_args: &[&str],
    timeout_secs: Option<f64>,
) -> Result<String, BridgeError> {
    let dir = tempfile::Builder::new()
        .prefix("noether-msolve-")
        .tempdir()
        .map_err(BridgeError::Spawn)?;
    let in_path = dir.path().join("in.ms");
    let out_path = dir.path().join("out.ms");
    let err_path = dir.path().join("err.txt");
    std::fs::write(&in_path, input_body).map_err(BridgeError::Spawn)?;
    let err_file = std::fs::File::create(&err_path).map_err(BridgeError::Spawn)?;

    let mut command = Command::new(MSOLVE_BIN);
    command
        .args(mode_args)
        .arg("-f")
        .arg(&in_path)
        .arg("-o")
        .arg(&out_path)
        // msolve writes the answer to `-o`; stdout is progress (discarded),
        // stderr -> a file so a long run never blocks on a full pipe.
        .stdout(Stdio::null())
        .stderr(Stdio::from(err_file));
    // The caller may need a private FLINT/GMP path for Noether itself. msolve
    // must instead resolve the native stack against which it was installed.
    isolate_native_libraries(&mut command);
    let mut child = command.spawn().map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            BridgeError::Unavailable(format!("`{MSOLVE_BIN}` not found on PATH"))
        } else {
            BridgeError::Spawn(e)
        }
    })?;

    let status = match timeout_secs {
        None => child.wait().map_err(BridgeError::Spawn)?,
        Some(secs) => {
            // `try_from_secs_f64` rejects negative / NaN / inf without panic;
            // a bad value collapses to ZERO → immediate timeout.
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
                        stderr: format!("msolve timed out after {secs}s"),
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
        .map_err(|e| BridgeError::Parse(format!("reading msolve output `{out_path:?}`: {e}")))
    // `dir` (and the temp files) are removed on drop here.
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A reduced Gröbner basis over F_2 of `<x^2 + y, y^2 + x>` via `-g 2`.
    /// Smoke test: msolve runs, writes `-o`, and the output mentions the
    /// variables. Skips when msolve is absent.
    #[test]
    fn msolve_groebner_smoke() {
        if !msolve_available() {
            eprintln!("skipping: `msolve` not on PATH");
            return;
        }
        // msolve input grammar: vars / characteristic / comma-separated polys.
        let body = "x,y\n2\nx^2+y,\ny^2+x\n";
        let out = run_msolve(body, &["-g", "2"], Some(60.0)).expect("msolve -g 2 should succeed");
        assert!(
            out.contains('x') || out.contains('y') || out.contains('1'),
            "unexpected msolve output: {out:?}"
        );
    }

    #[test]
    fn unavailable_is_typed_not_panic() {
        // The probe must never panic, regardless of environment.
        let _ = msolve_available();
    }
}
