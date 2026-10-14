// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use noether_bridges::BridgeError;

#[test]
fn unavailable_is_actionable_and_source_free() {
    let error = BridgeError::Unavailable("M2 not found on PATH".to_owned());
    assert_eq!(
        error.to_string(),
        "bridge unavailable: M2 not found on PATH"
    );
    assert!(std::error::Error::source(&error).is_none());
}

#[test]
fn subprocess_failure_preserves_status_and_stderr() {
    let error = BridgeError::NonZeroExit {
        code: Some(2),
        stderr: "bad input".to_owned(),
    };
    let rendered = error.to_string();
    assert!(rendered.contains("Some(2)"));
    assert!(rendered.contains("bad input"));
}
