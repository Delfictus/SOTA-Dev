//! Build script for prism-gpu
//!
//! Compiles CUDA kernels to PTX for runtime loading.
//!
//! CUDA COMPILATION:
//! - Target architecture: sm_120 (Blackwell GB202)
//! - Optimization level: -O3 (maximum performance)
//! - PTX output: target/ptx/<kernel_name>.ptx
//! - Optimized for Blackwell: 192 SM, 24,576 CUDA cores, 512 Tensor Cores (Gen 6)
//!
//! DEPENDENCIES:
//! - CUDA Toolkit 12.6 installed at /usr/local/cuda-12.6
//! - nvcc compiler in PATH or CUDA_HOME set
//!
//! SECURITY:
//! - PTX modules signed with SHA-256 for verification
//! - Signatures stored in target/ptx/<kernel_name>.ptx.sha256
//!
//! REFERENCE: PRISM GPU Plan §4.3 (PTX Compilation)

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/kernels/");

    // Check if CUDA feature is enabled
    let cuda_enabled = env::var("CARGO_FEATURE_CUDA").is_ok();
    if !cuda_enabled {
        println!("cargo:warning=CUDA feature not enabled, skipping PTX compilation");
        return;
    }

    // Link cuFFT library (part of CUDA toolkit, not an external dependency)
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=cufft");

    // Locate nvcc compiler
    let nvcc = find_nvcc().expect("nvcc not found. Ensure CUDA toolkit is installed.");

    println!("cargo:info=Using nvcc: {}", nvcc);

    // Create output directory
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx_dir = out_dir.join("ptx");
    std::fs::create_dir_all(&ptx_dir).expect("Failed to create PTX output directory");

    // Also create target/ptx for easier access.
    //
    // CRITICAL: cargo runs build scripts with CWD = the build script's crate
    // directory, NOT the workspace root. The previous version used a relative
    // path `target/ptx`, which (since this build.rs lives in
    // `crates/prism-gpu/`) resolved to `crates/prism-gpu/target/ptx/` rather
    // than the workspace `target/ptx/`. Meanwhile `find_twin_ptx` (in
    // `crates/prism-nhs/src/twin_kernels.rs`) loads PTX from the workspace
    // `target/ptx/` first. The result was that fresh PTX builds went into
    // a directory the engine never reads, and the vendored Apr-9 bundle
    // sitting in workspace `target/ptx/` "won" against every fresh build —
    // silently. Stage 2 was the symptom that exposed this: my kernel
    // changes built cleanly but produced zero observable effect because the
    // engine was loading the stale vendored kernel.
    //
    // Fix: anchor `target_ptx_dir` to `CARGO_MANIFEST_DIR/../../target/ptx`
    // so it always lands at the workspace root regardless of where cargo
    // runs the build script from.
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let target_ptx_dir = manifest_dir.join("../../target/ptx");
    std::fs::create_dir_all(&target_ptx_dir).expect("Failed to create target/ptx directory");

    // Compile Floyd-Warshall kernel
    compile_kernel(
        &nvcc,
        "src/kernels/floyd_warshall.cu",
        &ptx_dir.join("floyd_warshall.ptx"),
        &target_ptx_dir.join("floyd_warshall.ptx"),
    );

    // Compile Dendritic Reservoir kernel
    compile_kernel(
        &nvcc,
        "src/kernels/dendritic_reservoir.cu",
        &ptx_dir.join("dendritic_reservoir.ptx"),
        &target_ptx_dir.join("dendritic_reservoir.ptx"),
    );

    // Compile TDA kernel
    compile_kernel(
        &nvcc,
        "src/kernels/tda.cu",
        &ptx_dir.join("tda.ptx"),
        &target_ptx_dir.join("tda.ptx"),
    );

    // Compile Thermodynamic kernel (Phase 2)
    compile_kernel(
        &nvcc,
        "src/kernels/thermodynamic.cu",
        &ptx_dir.join("thermodynamic.ptx"),
        &target_ptx_dir.join("thermodynamic.ptx"),
    );

    // Compile Quantum kernel (Phase 3)
    compile_kernel(
        &nvcc,
        "src/kernels/quantum.cu",
        &ptx_dir.join("quantum.ptx"),
        &target_ptx_dir.join("quantum.ptx"),
    );

    // Compile Active Inference kernel (Phase 1)
    compile_kernel(
        &nvcc,
        "src/kernels/active_inference.cu",
        &ptx_dir.join("active_inference.ptx"),
        &target_ptx_dir.join("active_inference.ptx"),
    );

    // Compile PIMC kernel (Path Integral Monte Carlo for quantum annealing)
    compile_kernel(
        &nvcc,
        "src/kernels/pimc.cu",
        &ptx_dir.join("pimc.ptx"),
        &target_ptx_dir.join("pimc.ptx"),
    );

    // Compile Transfer Entropy kernel (KSG estimator for causal discovery)
    compile_kernel(
        &nvcc,
        "src/kernels/transfer_entropy.cu",
        &ptx_dir.join("transfer_entropy.ptx"),
        &target_ptx_dir.join("transfer_entropy.ptx"),
    );

    // Compile Ensemble Exchange kernel (CMA-ES replica management)
    compile_kernel(
        &nvcc,
        "src/kernels/ensemble_exchange.cu",
        &ptx_dir.join("ensemble_exchange.ptx"),
        &target_ptx_dir.join("ensemble_exchange.ptx"),
    );

    // Compile GNN Inference kernel (Graph Neural Network acceleration)
    compile_kernel(
        &nvcc,
        "src/kernels/gnn_inference.cu",
        &ptx_dir.join("gnn_inference.ptx"),
        &target_ptx_dir.join("gnn_inference.ptx"),
    );

    // Compile Molecular Dynamics kernel (MEC phase interactions)
    compile_kernel(
        &nvcc,
        "src/kernels/molecular_dynamics.cu",
        &ptx_dir.join("molecular_dynamics.ptx"),
        &target_ptx_dir.join("molecular_dynamics.ptx"),
    );

    // Compile CMA-ES kernel (Covariance Matrix Adaptation Evolution Strategy)
    compile_kernel(
        &nvcc,
        "src/kernels/cma_es.cu",
        &ptx_dir.join("cma_es.ptx"),
        &target_ptx_dir.join("cma_es.ptx"),
    );

    // Compile WHCR kernel (Wavelet-Hierarchical Conflict Repair)
    compile_kernel(
        &nvcc,
        "src/kernels/whcr.cu",
        &ptx_dir.join("whcr.ptx"),
        &target_ptx_dir.join("whcr.ptx"),
    );

    // Compile Dendritic WHCR kernel (Neuromorphic co-processor for conflict repair)
    compile_kernel(
        &nvcc,
        "src/kernels/dendritic_whcr.cu",
        &ptx_dir.join("dendritic_whcr.ptx"),
        &target_ptx_dir.join("dendritic_whcr.ptx"),
    );

    // Compile Dendritic SNN Reservoir kernel (Neuromorphic RL agent)
    compile_kernel(
        &nvcc,
        "src/kernels/dendritic_snn_reservoir.cu",
        &ptx_dir.join("dendritic_snn_reservoir.ptx"),
        &target_ptx_dir.join("dendritic_snn_reservoir.ptx"),
    );

    // Compile Gamma Envelope Reduction kernel (VASIL exact metric - PATH B)
    compile_kernel(
        &nvcc,
        "src/kernels/gamma_envelope_reduction.cu",
        &ptx_dir.join("gamma_envelope_reduction.ptx"),
        &target_ptx_dir.join("gamma_envelope_reduction.ptx"),
    );

    // Compile On-the-Fly P_neut kernel (memory-efficient VASIL - PATH B)
    compile_kernel(
        &nvcc,
        "src/kernels/prism_immunity_onthefly.cu",
        &ptx_dir.join("prism_immunity_onthefly.ptx"),
        &target_ptx_dir.join("prism_immunity_onthefly.ptx"),
    );

    // Compile Epitope-Based P_neut kernel (accuracy-optimized VASIL - PATH A)
    compile_kernel(
        &nvcc,
        "src/kernels/epitope_p_neut.cu",
        &ptx_dir.join("epitope_p_neut.ptx"),
        &target_ptx_dir.join("epitope_p_neut.ptx"),
    );

    // Compile Cryptic Hessian kernel (NiV-Bench Stage 12a)
    compile_kernel(
        &nvcc,
        "src/kernels/cryptic/cryptic_hessian.cu",
        &ptx_dir.join("cryptic_hessian.ptx"),
        &target_ptx_dir.join("cryptic_hessian.ptx"),
    );

    // Compile Cryptic Eigenmodes kernel (NiV-Bench Stage 12b)
    compile_kernel(
        &nvcc,
        "src/kernels/cryptic/cryptic_eigenmodes.cu",
        &ptx_dir.join("cryptic_eigenmodes.ptx"),
        &target_ptx_dir.join("cryptic_eigenmodes.ptx"),
    );

    // Compile Cryptic Probe Score kernel (NiV-Bench Stage 12c)
    compile_kernel(
        &nvcc,
        "src/kernels/cryptic/cryptic_probe_score.cu",
        &ptx_dir.join("cryptic_probe_score.ptx"),
        &target_ptx_dir.join("cryptic_probe_score.ptx"),
    );

    // Compile Cryptic Signal Fusion kernel (NiV-Bench Stage 12d)
    compile_kernel(
        &nvcc,
        "src/kernels/cryptic/cryptic_signal_fusion.cu",
        &ptx_dir.join("cryptic_signal_fusion.ptx"),
        &target_ptx_dir.join("cryptic_signal_fusion.ptx"),
    );

    // Compile Feature Merge kernel (Phase 1.5)
    compile_kernel(
        &nvcc,
        "src/kernels/feature_merge.cu",
        &ptx_dir.join("feature_merge.ptx"),
        &target_ptx_dir.join("feature_merge.ptx"),
    );

    // NOTE: Viral Evolution Fitness kernel disabled - fitness+cycle integrated into mega_fused Stages 7-8
    // compile_kernel(
    //     &nvcc,
    //     "src/kernels/viral_evolution_fitness.cu",
    //     &ptx_dir.join("viral_evolution_fitness.ptx"),
    //     &target_ptx_dir.join("viral_evolution_fitness.ptx"),
    // );

    // Compile PRISM-NOVA kernel (Hamiltonian Monte Carlo + Active Inference physics engine)
    compile_kernel(
        &nvcc,
        "src/kernels/prism_nova.cu",
        &ptx_dir.join("prism_nova.ptx"),
        &target_ptx_dir.join("prism_nova.ptx"),
    );

    // Compile AMBER Bonded Forces kernel (ff14SB force field - bonds, angles, dihedrals, 1-4)
    compile_kernel(
        &nvcc,
        "src/kernels/amber_bonded.cu",
        &ptx_dir.join("amber_bonded.ptx"),
        &target_ptx_dir.join("amber_bonded.ptx"),
    );

    // Compile AMBER Mega-Fused HMC kernel (full AMBER ff14SB + BAOAB Langevin thermostat)
    compile_kernel(
        &nvcc,
        "src/kernels/amber_mega_fused.cu",
        &ptx_dir.join("amber_mega_fused.ptx"),
        &target_ptx_dir.join("amber_mega_fused.ptx"),
    );

    // Compile SIMD Batch kernel (Tier 1: 10-50x throughput, identical physics)
    compile_kernel(
        &nvcc,
        "src/kernels/amber_simd_batch.cu",
        &ptx_dir.join("amber_simd_batch.ptx"),
        &target_ptx_dir.join("amber_simd_batch.ptx"),
    );

    // Compile PME kernel (Particle Mesh Ewald long-range electrostatics)
    compile_kernel(
        &nvcc,
        "src/kernels/pme.cu",
        &ptx_dir.join("pme.ptx"),
        &target_ptx_dir.join("pme.ptx"),
    );

    // Compile SETTLE kernel (rigid water constraint solver)
    compile_kernel(
        &nvcc,
        "src/kernels/settle.cu",
        &ptx_dir.join("settle.ptx"),
        &target_ptx_dir.join("settle.ptx"),
    );

    // Compile H-bond Constraints kernel (analytic SHAKE/RATTLE for protein X-H bonds)
    compile_kernel(
        &nvcc,
        "src/kernels/h_constraints.cu",
        &ptx_dir.join("h_constraints.ptx"),
        &target_ptx_dir.join("h_constraints.ptx"),
    );

    // Compile Verlet List kernel (SOTA optimization: 2-3x speedup for non-bonded)
    compile_kernel(
        &nvcc,
        "src/kernels/verlet_list.cu",
        &ptx_dir.join("verlet_list.ptx"),
        &target_ptx_dir.join("verlet_list.ptx"),
    );

    // Compile Tensor Core Forces kernel (SOTA optimization: 2-4x speedup with WMMA)
    compile_kernel(
        &nvcc,
        "src/kernels/tensor_core_forces.cu",
        &ptx_dir.join("tensor_core_forces.ptx"),
        &target_ptx_dir.join("tensor_core_forces.ptx"),
    );

    // Compile LCPO SASA kernel (GPU-accelerated solvent accessible surface area)
    // Used by cryptic site detection pipeline for fast per-frame SASA calculation
    compile_kernel(
        &nvcc,
        "src/kernels/lcpo_sasa.cu",
        &ptx_dir.join("lcpo_sasa.ptx"),
        &target_ptx_dir.join("lcpo_sasa.ptx"),
    );

    // =========================================================================
    // NHS (Neuromorphic Holographic Stream) Kernels
    // =========================================================================

    // Compile NHS Exclusion kernel (Hydrophobic Exclusion Mapping)
    // 30,000× faster than explicit solvent via "holographic negative" principle
    compile_kernel(
        &nvcc,
        "src/kernels/nhs_exclusion.cu",
        &ptx_dir.join("nhs_exclusion.ptx"),
        &target_ptx_dir.join("nhs_exclusion.ptx"),
    );

    // =========================================================================
    // CRITICAL: NHS-AMBER FUSED ENGINE (PRIMARY PRODUCTION KERNEL)
    // =========================================================================
    // This is THE MOST IMPORTANT kernel - handles full MD simulation + cryptic detection
    compile_kernel(
        &nvcc,
        "src/kernels/nhs_amber_fused.cu",
        &ptx_dir.join("nhs_amber_fused.ptx"),
        &target_ptx_dir.join("nhs_amber_fused.ptx"),
    );

    // Compile NHS Neuromorphic kernel (LIF dewetting detection network)
    // Spike-based detection of cryptic pocket opening events
    compile_kernel(
        &nvcc,
        "src/kernels/nhs_neuromorphic.cu",
        &ptx_dir.join("nhs_neuromorphic.ptx"),
        &target_ptx_dir.join("nhs_neuromorphic.ptx"),
    );

    // =========================================================================
    // ULTIMATE HYPEROPTIMIZED KERNELS (2-4x PERFORMANCE BOOST)
    // =========================================================================

    // Compile Ultimate MD kernel (14 GPU optimizations for maximum performance)
    // Expected speedup: 2-4x over standard kernel on Blackwell/Ampere/Ada
    compile_kernel(
        &nvcc,
        "src/kernels/ultimate_md.cu",
        &ptx_dir.join("ultimate_md.ptx"),
        &target_ptx_dir.join("ultimate_md.ptx"),
    );

    // Compile Hyperoptimized MD kernel (alternative optimization strategy)
    compile_kernel(
        &nvcc,
        "src/kernels/hyperoptimized_md.cu",
        &ptx_dir.join("hyperoptimized_md.ptx"),
        &target_ptx_dir.join("hyperoptimized_md.ptx"),
    );

    // Compile Pharmacophore Gaussian splatting kernel
    // Replaces Python pharmacophore_extract.py with GPU-accelerated density mapping
    compile_kernel(
        &nvcc,
        "src/kernels/pharmacophore_splat.cu",
        &ptx_dir.join("pharmacophore_splat.ptx"),
        &target_ptx_dir.join("pharmacophore_splat.ptx"),
    );

    // =========================================================================
    // PRISM-TWIN Kernels
    // =========================================================================

    // Protocol Director (TWIN v3.0 Gate 0: VRAM-resident protocol state machine)
    compile_kernel(
        &nvcc,
        "src/kernels/protocol_director.cu",
        &ptx_dir.join("protocol_director.ptx"),
        &target_ptx_dir.join("protocol_director.ptx"),
    );

    // Housekeeping kernels (TWIN v3.0 Gate 2: GPU-side COM removal, CA restraints, heartbeat)
    compile_kernel(
        &nvcc,
        "src/kernels/housekeeping.cu",
        &ptx_dir.join("housekeeping.ptx"),
        &target_ptx_dir.join("housekeeping.ptx"),
    );

    // Stream completion signaling (standard kernel — no cooperative groups)
    compile_kernel(
        &nvcc,
        "src/kernels/twin_signal.cu",
        &ptx_dir.join("twin_signal.ptx"),
        &target_ptx_dir.join("twin_signal.ptx"),
    );

    // Ring Buffer (standard kernel — no cooperative groups)
    compile_kernel(
        &nvcc,
        "src/kernels/ring_buffer.cu",
        &ptx_dir.join("ring_buffer.ptx"),
        &target_ptx_dir.join("ring_buffer.ptx"),
    );

    // Transfer Entropy (standard kernel — binned TE, no cooperative groups)
    compile_kernel(
        &nvcc,
        "src/kernels/twin_transfer_entropy.cu",
        &ptx_dir.join("twin_transfer_entropy.ptx"),
        &target_ptx_dir.join("twin_transfer_entropy.ptx"),
    );

    // Tensor Core CCF (standard kernel — WMMA, no cooperative groups)
    compile_kernel(
        &nvcc,
        "src/kernels/tensor_ccf.cu",
        &ptx_dir.join("tensor_ccf.ptx"),
        &target_ptx_dir.join("tensor_ccf.ptx"),
    );

    // TWIN Persistent Coupling (cooperative groups — the production coupling kernel)
    compile_cooperative_kernel(
        &nvcc,
        "src/kernels/twin_coupling_persistent.cu",
        &ptx_dir.join("twin_coupling_persistent.ptx"),
        &target_ptx_dir.join("twin_coupling_persistent.ptx"),
    );

    // TWIN Persistent Skeleton (cooperative groups — requires -rdc=true)
    compile_cooperative_kernel(
        &nvcc,
        "src/kernels/twin_persistent.cu",
        &ptx_dir.join("twin_persistent.ptx"),
        &target_ptx_dir.join("twin_persistent.ptx"),
    );

    // TWIN Persistent Physics (cooperative groups — requires -rdc=true)
    compile_cooperative_kernel(
        &nvcc,
        "src/kernels/twin_persistent_physics.cu",
        &ptx_dir.join("twin_persistent_physics.ptx"),
        &target_ptx_dir.join("twin_persistent_physics.ptx"),
    );

    println!("cargo:info=PTX compilation completed successfully");
}

/// Finds nvcc compiler in PATH or CUDA_HOME
fn find_nvcc() -> Option<String> {
    // Check CUDA_HOME environment variable
    if let Ok(cuda_home) = env::var("CUDA_HOME") {
        let nvcc_path = PathBuf::from(cuda_home).join("bin").join("nvcc");
        if nvcc_path.exists() {
            return Some(nvcc_path.to_string_lossy().to_string());
        }
    }

    // Check common CUDA installation paths
    let common_paths = vec![
        "/usr/local/cuda/bin/nvcc",
        "/usr/local/cuda-12.6/bin/nvcc",
        "/usr/local/cuda-12/bin/nvcc",
        "/opt/cuda/bin/nvcc",
    ];

    for path in common_paths {
        if PathBuf::from(path).exists() {
            return Some(path.to_string());
        }
    }

    // Check if nvcc is in PATH
    if Command::new("nvcc").arg("--version").output().is_ok() {
        return Some("nvcc".to_string());
    }

    None
}

/// Compiles a CUDA kernel to PTX
///
/// # Arguments
/// * `nvcc` - Path to nvcc compiler
/// * `source` - Path to .cu source file
/// * `output` - Path to output .ptx file (in OUT_DIR)
/// * `target_output` - Path to copy .ptx file (in target/ptx)
fn compile_kernel(nvcc: &str, source: &str, output: &PathBuf, target_output: &PathBuf) {
    println!("cargo:info=Compiling {} -> {}", source, output.display());

    let status = Command::new(nvcc)
        .arg("--ptx") // Generate PTX (not binary)
        .arg("-o")
        .arg(output)
        .arg(source)
        // Target architecture (sm_120 = Blackwell GB202)
        // Note: --ptx mode only supports single architecture
        .arg("-arch=sm_120")
        // Optimization flags
        .arg("-O3") // Maximum optimization
        // ── Phase 0 (Bug C fix): --use_fast_math intentionally REMOVED ──
        //
        // --use_fast_math implies:
        //   • --ftz=true       (flush denormals to zero)
        //   • --prec-div=false (approximate division)
        //   • --prec-sqrt=false (approximate sqrt)
        //   • --fmad=true      (fused multiply-add)
        //
        // These flags caused PTX codegen drift between freshly-built and
        // vendored PTX bundles. Specifically, `protocol_director.ptx` from
        // a fresh build with --use_fast_math used `div.approx.ftz.f32` /
        // `sub.ftz.f32` / `fma.rn.ftz.f32` instead of the IEEE-correct
        // `div.rn.f32` / `sub.f32` / `fma.rn.f32` opcodes used in the
        // vendored bundle. When the engine computed small phase deltas
        // near zero, FTZ flushed them to zero → division produced inf or
        // NaN → integer cast → out-of-bounds index → SIGSEGV in the
        // Director kernel.
        //
        // The bootstrap commit (24c6cdb1) worked around this by vendoring
        // a known-good PTX bundle in vendor/working_ptx_2026-04-10/ and
        // having scripts/prism-validate-and-run.sh seed target/ptx/ from
        // it when freshly-built protocol_director.ptx was missing.
        //
        // Removing --use_fast_math here makes fresh builds match the
        // vendored bundle's IEEE semantics. The vendored bundle becomes
        // unnecessary and can be retired in a follow-up commit once this
        // change is validated against the bootstrap signature gate.
        //
        // Performance impact: minimal. PRISM-4D's hot kernels (Physics,
        // multi_lif) are dominated by memory bandwidth, not transcendental
        // throughput. The IEEE-correct division and sub instructions cost
        // ~5% more cycles than their fast-math counterparts but the
        // bottleneck is L2/HBM, not the ALU.
        //
        // If a future kernel genuinely needs fast-math semantics for
        // performance, it should use the --use_fast_math equivalent
        // intrinsics directly (`__fdividef`, `__fadd_rz`, etc.) on a
        // per-call-site basis rather than enabling it globally.
        .arg("--restrict") // Enable restrict keyword optimization
        // Include paths
        .arg("-I/usr/local/cuda/include")
        .arg("-Isrc/kernels")  // For shared headers (protocol_state.cuh)
        // Warning flags
        .arg("-Xptxas=-v") // Verbose PTX assembly info (shows register usage)
        .arg("--expt-relaxed-constexpr") // Allow constexpr in device code
        .status()
        .expect("Failed to execute nvcc");

    if !status.success() {
        panic!("nvcc compilation failed for {}", source);
    }

    // Copy PTX to target/ptx for easy access
    std::fs::copy(output, target_output).expect("Failed to copy PTX to target/ptx");

    // Generate SHA-256 signature for security verification
    generate_ptx_signature(target_output);

    println!("cargo:info=PTX compiled: {}", target_output.display());
}

/// Compiles a CUDA cooperative kernel to PTX (requires -rdc=true for grid.sync())
fn compile_cooperative_kernel(nvcc: &str, source: &str, output: &PathBuf, target_output: &PathBuf) {
    println!("cargo:info=Compiling (cooperative) {} -> {}", source, output.display());

    let status = Command::new(nvcc)
        .arg("--ptx")
        .arg("-o")
        .arg(output)
        .arg(source)
        .arg("-arch=sm_120")
        .arg("-rdc=true")           // relocatable device code for grid.sync()
        .arg("-O3")
        // --use_fast_math intentionally REMOVED — see compile_kernel
        // above for the full rationale (Phase 0 / Bug C fix). Same fix
        // applies here for cooperative kernels (twin_persistent_physics
        // and friends): IEEE-correct codegen prevents the PTX drift
        // that caused SIGSEGV when small phase deltas got flushed to
        // zero by FTZ semantics.
        .arg("--restrict")
        .arg("-I/usr/local/cuda/include")
        .arg("-Isrc/kernels")  // For shared headers (protocol_state.cuh)
        .arg("-Xptxas=-v")
        .arg("--expt-relaxed-constexpr")
        .status()
        .expect("Failed to execute nvcc");

    if !status.success() {
        panic!("nvcc cooperative compilation failed for {}", source);
    }

    std::fs::copy(output, target_output).expect("Failed to copy cooperative PTX to target/ptx");
    generate_ptx_signature(target_output);
    println!("cargo:info=PTX compiled (cooperative): {}", target_output.display());
}

/// Generates SHA-256 signature for PTX file
///
/// Signature is stored in <ptx_file>.sha256 for runtime verification.
fn generate_ptx_signature(ptx_path: &PathBuf) {
    use sha2::{Digest, Sha256};

    let ptx_bytes = std::fs::read(ptx_path).expect("Failed to read PTX file");
    let hash = Sha256::digest(&ptx_bytes);
    let hash_hex = hex::encode(hash);

    let sig_path = ptx_path.with_extension("ptx.sha256");
    std::fs::write(&sig_path, hash_hex).expect("Failed to write signature file");

    println!("cargo:info=PTX signature: {}", sig_path.display());
}
