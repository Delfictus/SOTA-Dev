# fpocket Integration for PRISM-LBS

## Overview

This integration allows PRISM-LBS to leverage **fpocket**, the gold-standard open-source tool for protein pocket detection, as an alternative to internal algorithms.

fpocket is a widely-validated, publication-quality pocket detection algorithm based on Voronoi tessellation and alpha spheres. By integrating fpocket via FFI, PRISM-LBS can provide:

- **Validated Results**: fpocket is used in hundreds of publications
- **High Accuracy**: Proven performance on PDBBind, DUD-E, and other benchmarks
- **Druggability Scores**: Built-in druggability estimation
- **Fast Execution**: Optimized C implementation

## Reference

Le Guilloux, V., Schmidtke, P., & Tuffery, P. (2009). "Fpocket: An open source platform for ligand pocket detection." *BMC Bioinformatics*, 10(1), 168.

## Installation

### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install fpocket
```

### macOS
```bash
brew install fpocket
```

### From Source
```bash
git clone https://github.com/Discngine/fpocket.git
cd fpocket
make
sudo make install
```

### Verify Installation
```bash
which fpocket
fpocket -h
```

## Usage

### Option 1: Programmatic API

```rust
use prism_lbs::pocket::fpocket_ffi::{fpocket_available, run_fpocket, FpocketConfig};
use std::path::Path;

// Check if fpocket is available
if !fpocket_available() {
    eprintln!("fpocket not found. Please install fpocket.");
    return;
}

// Configure fpocket
let config = FpocketConfig {
    min_alpha_radius: 3.0,           // Minimum pocket radius (Angstroms)
    max_pockets: 20,                 // Maximum number of pockets to return
    druggability_threshold: 0.5,     // Minimum druggability score (0.0-1.0)
    ..Default::default()
};

// Run fpocket on PDB file
let pdb_path = Path::new("data/1hiv.pdb");
let pockets = run_fpocket(&pdb_path, &config)?;

// Process results
for (i, pocket) in pockets.iter().enumerate() {
    println!("Pocket {}: druggability = {:.3}, volume = {:.1} Ų",
             i + 1,
             pocket.druggability_score.total,
             pocket.volume);
}
```

### Option 2: Integrated Detector

```rust
use prism_lbs::{LbsConfig, PocketDetector, ProteinStructure};
use std::path::Path;

// Load PDB structure (must use from_pdb_file to enable fpocket)
let pdb_path = Path::new("data/1hiv.pdb");
let structure = ProteinStructure::from_pdb_file(pdb_path)?;
let graph = ProteinGraphBuilder::new().build(&structure)?;

// Configure detector to use fpocket
let mut config = LbsConfig::default();
config.use_fpocket = true;  // Enable fpocket integration

let detector = PocketDetector::new(config)?;
let pockets = detector.detect(&graph)?;
```

### Option 3: CLI (if implemented)

```bash
prism-lbs detect --fpocket input.pdb
```

## Detection Priority

PRISM-LBS uses a cascading detection strategy:

1. **Priority 1: fpocket** (if enabled and available)
   - Gold-standard Voronoi-based detection
   - Requires `use_fpocket = true` and fpocket installed
   - Requires PDB loaded via `from_pdb_file()` (not `from_pdb_str()`)

2. **Priority 2: Voronoi detector** (if enabled)
   - Internal Delaunay triangulation algorithm
   - Requires `use_voronoi_detection = true`

3. **Priority 3: Legacy cavity detector**
   - Grid-based alpha sphere detection
   - Requires `use_cavity_detection = true`

4. **Priority 4: Belief propagation** (fallback)
   - Graph coloring approach
   - Always available

## Configuration

### FpocketConfig

```rust
pub struct FpocketConfig {
    /// Integration mode (Binary or NativeLib)
    pub mode: FpocketMode,

    /// Path to fpocket binary (optional, searches PATH by default)
    pub fpocket_path: Option<PathBuf>,

    /// Minimum alpha sphere radius in Angstroms
    /// Default: 3.0 (drug-bindable cavities)
    pub min_alpha_radius: f64,

    /// Maximum number of pockets to return
    /// Default: 20
    pub max_pockets: usize,

    /// Druggability score threshold (0.0-1.0)
    /// Default: 0.0 (no filtering)
    pub druggability_threshold: f64,
}
```

### PocketDetectorConfig

```rust
pub struct PocketDetectorConfig {
    /// Enable fpocket integration
    pub use_fpocket: bool,

    /// fpocket configuration
    pub fpocket: FpocketConfig,

    // ... other detection methods
}
```

## fpocket Output Format

fpocket creates an output directory `<input>_out/` containing:

### Files Created
- `<input>_info.txt` - Pocket statistics (volume, druggability, SASA, etc.)
- `<input>_pockets.pdb` - All pockets visualized as spheres
- `pockets/pocket<N>_atm.pdb` - Individual pocket atoms
- `pockets/pocket<N>_vert.pqr` - Voronoi vertices (alpha spheres)

### Parsed Properties

The FFI integration extracts:
- **Volume**: Pocket volume (ų)
- **Druggability**: fpocket druggability score (0.0-1.0)
- **SASA**: Solvent-accessible surface area (ų)
- **Hydrophobicity**: Mean local hydrophobic density
- **Alpha Spheres**: Number of alpha spheres forming the pocket
- **Centroid**: Geometric center of pocket atoms
- **Atom Indices**: Protein atoms in contact with pocket
- **Residue Indices**: Residues forming the pocket

## Druggability Classification

fpocket scores (0.0-1.0) are mapped to PRISM classifications:

| fpocket Score | PRISM Class      | Interpretation                    |
|---------------|------------------|-----------------------------------|
| ≥ 0.7         | HighlyDruggable  | Excellent drug target             |
| 0.5 - 0.7     | Druggable        | Good drug target                  |
| 0.3 - 0.5     | DifficultTarget  | Challenging but possible          |
| < 0.3         | Undruggable      | Not suitable for small molecules  |

## Example: HIV-1 Protease

```rust
use prism_lbs::pocket::fpocket_ffi::{run_fpocket, FpocketConfig};
use std::path::Path;

let pdb = Path::new("data/1hiv.pdb");
let config = FpocketConfig::default();
let pockets = run_fpocket(&pdb, &config)?;

// Expected: Active site pocket ~400-600 Ų, druggability > 0.7
let active_site = &pockets[0];
println!("Active site:");
println!("  Volume: {:.1} Ų", active_site.volume);  // ~500 Ų
println!("  Druggability: {:.3}", active_site.druggability_score.total);  // ~0.8
```

## Performance

fpocket is typically **faster** than internal methods:

| Method                | Time (1hiv.pdb, 198 residues) |
|-----------------------|-------------------------------|
| fpocket (C binary)    | ~0.5-1.0 seconds              |
| Internal Voronoi      | ~2-5 seconds                  |
| Grid cavity detector  | ~3-8 seconds                  |
| Belief propagation    | ~5-15 seconds                 |

## Limitations

1. **Binary Dependency**: Requires fpocket installation
2. **File I/O**: Requires PDB file on disk (not in-memory string)
3. **External Process**: Subprocess overhead (~50-100ms)
4. **Output Parsing**: Relies on fpocket output format stability

## Troubleshooting

### "fpocket not found in PATH"

**Solution**: Install fpocket and ensure it's in your PATH:
```bash
which fpocket  # Should return path to fpocket binary
export PATH=$PATH:/path/to/fpocket/bin
```

### "fpocket enabled but no PDB file path available"

**Solution**: Use `from_pdb_file()` instead of `from_pdb_str()`:
```rust
// ❌ Won't work with fpocket
let structure = ProteinStructure::from_pdb_str(&pdb_string)?;

// ✅ Works with fpocket
let structure = ProteinStructure::from_pdb_file(Path::new("input.pdb"))?;
```

### "fpocket execution failed"

**Causes**:
- Invalid PDB file
- fpocket version incompatibility
- Permission issues

**Debug**:
```bash
# Test fpocket manually
fpocket -f input.pdb

# Check output directory
ls input_out/
```

## Feature Flag

The fpocket integration is optional and can be disabled:

```toml
# Cargo.toml
[dependencies]
prism-lbs = { version = "0.3", features = ["cuda", "fpocket"] }
```

```rust
#[cfg(feature = "fpocket")]
use prism_lbs::pocket::fpocket_ffi::run_fpocket;
```

## Testing

```bash
# Run fpocket integration tests (requires fpocket installed)
cargo test --features fpocket fpocket

# Run example
cargo run --example fpocket_demo --features fpocket -- data/1hiv.pdb
```

## Future Enhancements

1. **Native Library Binding**: Link to `libfpocket.so` instead of subprocess
2. **Custom fpocket Parameters**: Expose more fpocket configuration options
3. **Streaming Parser**: Parse fpocket output incrementally
4. **Batch Processing**: Run fpocket on multiple structures in parallel

## References

- fpocket Website: https://github.com/Discngine/fpocket
- fpocket Paper: https://doi.org/10.1186/1471-2105-10-168
- PRISM-LBS Documentation: https://github.com/prism-ai/prism-lbs
