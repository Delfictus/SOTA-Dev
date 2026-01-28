//! CI Guardrail: No Synthetic Events
//!
//! This test enforces the HARD INVARIANT that no synthetic/placeholder/mock
//! event generation exists in the production codebase.
//!
//! If this test fails, someone has added code that can fabricate events.
//! This is a FATAL violation of the data integrity contract.

use std::fs;
use std::path::Path;

/// Banned tokens that indicate synthetic event generation.
/// These MUST NOT appear in production source code.
///
/// Note: "placeholder" alone is OK (for placeholder values/structures).
/// Only phrases that indicate FABRICATED EVENTS or PLACEHOLDER METRICS are banned.
const BANNED_TOKENS: &[&str] = &[
    // Synthetic event generation
    "synthetic event",
    "synthetic_event",
    "generate_synthetic",
    "generate synthetic",
    // Mock/fake events
    "mock event",
    "mock_event",
    "fake event",
    "fake_event",
    "fabricated event",
    // Demo/test event generation (in production code)
    "demo event",
    "demo_event",
    "dummy event",
    "dummy_event",
    // Explicit fallback paths
    "fallback event",
    "fallback_event",
];

/// Additional context-sensitive patterns that are forbidden
/// These indicate code that generates fake data for testing
const BANNED_PATTERNS: &[&str] = &[
    // Phrases that indicate synthetic data generation
    "generating synthetic events",
    "generate fake events",
    "for testing the pipeline",
    "will be replaced with actual",
    // Placeholder metrics (FORBIDDEN - must compute or emit null)
    "placeholder residue",
    "// placeholder",
    "placeholder results",
    "generate placeholder sites",
];

/// Scan a file for banned tokens
fn scan_file_for_banned_tokens(path: &Path) -> Vec<(usize, String, String)> {
    let mut violations = Vec::new();

    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return violations,
    };

    let content_lower = content.to_lowercase();

    for (line_num, line) in content.lines().enumerate() {
        let line_lower = line.to_lowercase();

        // Check banned tokens
        for token in BANNED_TOKENS {
            if line_lower.contains(token) {
                violations.push((
                    line_num + 1,
                    token.to_string(),
                    line.trim().to_string(),
                ));
            }
        }

        // Check banned patterns
        for pattern in BANNED_PATTERNS {
            if line_lower.contains(&pattern.to_lowercase()) {
                violations.push((
                    line_num + 1,
                    pattern.to_string(),
                    line.trim().to_string(),
                ));
            }
        }
    }

    violations
}

/// Recursively scan a directory for banned tokens
fn scan_directory(dir: &Path, violations: &mut Vec<(String, usize, String, String)>) {
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let path = entry.path();

        // Skip target directory
        if path.ends_with("target") || path.to_string_lossy().contains("/target/") {
            continue;
        }

        // Skip test files - they're allowed to have test data
        let path_str = path.to_string_lossy();
        if path_str.contains("/tests/") || path_str.ends_with("_test.rs") {
            continue;
        }

        // Skip this test file itself
        if path_str.ends_with("no_synthetic_events_test.rs") {
            continue;
        }

        if path.is_dir() {
            scan_directory(&path, violations);
        } else if path.extension().map(|e| e == "rs").unwrap_or(false) {
            let file_violations = scan_file_for_banned_tokens(&path);
            for (line, token, context) in file_violations {
                violations.push((
                    path.to_string_lossy().to_string(),
                    line,
                    token,
                    context,
                ));
            }
        }
    }
}

#[test]
fn test_no_banned_tokens_in_source() {
    let src_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");

    let mut violations = Vec::new();
    scan_directory(&src_dir, &mut violations);

    if !violations.is_empty() {
        let mut msg = String::from("\n");
        msg.push_str("╔════════════════════════════════════════════════════════════════╗\n");
        msg.push_str("║  FATAL: BANNED TOKENS FOUND IN SOURCE CODE                     ║\n");
        msg.push_str("║                                                                ║\n");
        msg.push_str("║  The following violations indicate code that can fabricate     ║\n");
        msg.push_str("║  synthetic events. This is FORBIDDEN in production builds.     ║\n");
        msg.push_str("╚════════════════════════════════════════════════════════════════╝\n\n");

        for (file, line, token, context) in &violations {
            msg.push_str(&format!("  {}:{}\n", file, line));
            msg.push_str(&format!("    Token: '{}'\n", token));
            msg.push_str(&format!("    Context: {}\n\n", context));
        }

        msg.push_str("ACTION REQUIRED:\n");
        msg.push_str("  1. Remove ALL synthetic event generation code\n");
        msg.push_str("  2. Ensure prism4d run ONLY uses real engine output\n");
        msg.push_str("  3. There is NO fallback path - real events required\n");

        panic!("{}", msg);
    }
}

#[test]
fn test_run_engine_requires_gpu_feature() {
    // Verify that run_cryo_uv_engine is properly gated behind the gpu feature
    // by checking the source code
    let prism4d_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("bin")
        .join("prism4d.rs");

    let content = fs::read_to_string(&prism4d_path)
        .expect("Failed to read prism4d.rs");

    // Check that the non-gpu version bails
    assert!(
        content.contains("#[cfg(not(feature = \"gpu\"))]") &&
        content.contains("bail!("),
        "Non-GPU run_cryo_uv_engine must bail with clear error message"
    );

    // Check that validate_events_file is called
    assert!(
        content.contains("validate_events_file"),
        "Pipeline must validate events.jsonl"
    );

    // Check there's no fallback generation
    assert!(
        !content.contains("generate_synthetic") &&
        !content.contains("synthetic_event") &&
        !content.contains("for testing the pipeline"),
        "No synthetic event generation allowed"
    );
}

#[test]
fn test_validate_events_file_exists() {
    // Verify that the validation function exists and has proper checks
    let prism4d_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("bin")
        .join("prism4d.rs");

    let content = fs::read_to_string(&prism4d_path)
        .expect("Failed to read prism4d.rs");

    // Must check file exists
    assert!(
        content.contains("events_path.exists()"),
        "Must check events file exists"
    );

    // Must check file is non-empty
    assert!(
        content.contains("metadata.len() == 0") || content.contains("metadata.len() == 0"),
        "Must check events file is non-empty"
    );

    // Must validate JSON structure
    assert!(
        content.contains("serde_json::from_str"),
        "Must validate JSON structure"
    );

    // Must check required fields
    assert!(
        content.contains("center_xyz") && content.contains("volume_a3") && content.contains("phase"),
        "Must validate required event fields"
    );
}

#[test]
fn test_no_alternate_execution_paths() {
    // Ensure there's no way to bypass the real engine
    let prism4d_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("bin")
        .join("prism4d.rs");

    let content = fs::read_to_string(&prism4d_path)
        .expect("Failed to read prism4d.rs");

    // Must not have legacy/placeholder/demo mode flags
    let forbidden_args = [
        "--legacy",
        "--placeholder",
        "--demo",
        "--synthetic",
        "--fake",
        "--mock",
        "--test-mode",
    ];

    for arg in &forbidden_args {
        assert!(
            !content.contains(arg),
            "Forbidden argument '{}' found in CLI. No alternate execution paths allowed.",
            arg
        );
    }
}
