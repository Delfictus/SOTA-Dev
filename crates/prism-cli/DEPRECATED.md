# DEPRECATED: prism-cli

> **This crate has been fully deprecated and excluded from the workspace.**

## Migration Status: COMPLETE

All functionality from `prism-cli` has been migrated to the unified TUI at `crates/prism/`.

## What Was Migrated

| Component | Original Location | New Location |
|-----------|-------------------|--------------|
| `config.rs` | `prism-cli/src/config.rs` | `prism/src/config.rs` |
| `metrics_server.rs` | `prism-cli/src/metrics_server.rs` | `prism/src/metrics_server.rs` |
| CLI arguments | `prism-cli/src/main.rs` | `prism/src/main.rs` |
| Modes (coloring, biomolecular, materials, mec-only) | `prism-cli/src/main.rs` | Integrated into unified TUI |

## New Entry Point

The unified PRISM interface is now at:

```bash
# Default (world-class TUI)
cargo run --release

# Or explicitly
cargo run --release -p prism

# Headless mode (equivalent to old prism-cli)
cargo run --release -- --headless --input graph.col
```

## Why Deprecated?

1. **Single Entry Point**: Having two binaries (`prism` and `prism-cli`) caused confusion
2. **Code Duplication**: Both binaries shared 90%+ identical code
3. **Maintenance Burden**: Updates had to be applied to both interfaces
4. **User Experience**: The unified TUI provides superior visualization and interaction

## Timeline

- **v0.2.x**: Both `prism` and `prism-cli` coexisted
- **v0.3.0**: `prism-cli` deprecated, excluded from workspace
- **v0.4.0**: `prism-cli` directory will be removed entirely

## Archive

This directory is preserved temporarily for reference. It will be removed in a future release.

---

*Last updated: 2024-11-29*
*Migration completed by: Claude Code*
