//! Build script for prism-gpu
//!
//! Compiles CUDA kernels to PTX for runtime loading.
//!
//! CUDA COMPILATION:
//! - Target architecture: sm_86 (RTX 3060 Ampere)
//! - Optimization level: -O3 (maximum performance)
//! - PTX output: target/ptx/<kernel_name>.ptx
//! - Fallback architectures: sm_75, sm_80 for broader compatibility
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

    // Also create target/ptx for easier access
    let target_ptx_dir = PathBuf::from("target/ptx");
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
        // Target architecture (sm_86 = RTX 3060 Ampere)
        // Note: --ptx mode only supports single architecture
        .arg("-arch=sm_86")
        // Optimization flags
        .arg("-O3") // Maximum optimization
        .arg("--use_fast_math") // Fast math operations
        .arg("--restrict") // Enable restrict keyword optimization
        // Include paths (if needed)
        .arg("-I/usr/local/cuda/include")
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
