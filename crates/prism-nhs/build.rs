//! Build script for prism-nhs
//!
//! Compiles CUDA kernels to PTX when the `gpu` feature is enabled.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/cuda/spike_density.cu");

    let gpu_enabled = env::var("CARGO_FEATURE_GPU").is_ok();
    if !gpu_enabled {
        return;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let nvcc = find_nvcc().expect(
        "nvcc not found. CUDA toolkit is required for the `gpu` feature.\n\
         Set CUDA_HOME or ensure nvcc is in PATH.",
    );

    compile_kernel(&nvcc, "src/cuda/spike_density.cu", &out_dir.join("spike_density.ptx"));
}

fn compile_kernel(nvcc: &str, source: &str, output: &PathBuf) {
    println!("cargo:info=Compiling {} -> {}", source, output.display());

    let status = Command::new(nvcc)
        .arg("--ptx")
        .arg("-o")
        .arg(output)
        .arg(source)
        .arg("-arch=sm_120") // Blackwell GB202 (RTX 5080)
        .arg("-O3")
        .arg("--use_fast_math")
        .arg("--restrict")
        .arg("-I/usr/local/cuda/include")
        .status()
        .expect("Failed to execute nvcc");

    if !status.success() {
        panic!("nvcc compilation failed for {}", source);
    }

    println!("cargo:info=PTX compiled: {}", output.display());
}

fn find_nvcc() -> Option<String> {
    if let Ok(cuda_home) = env::var("CUDA_HOME") {
        let p = PathBuf::from(&cuda_home).join("bin").join("nvcc");
        if p.exists() {
            return Some(p.to_string_lossy().to_string());
        }
    }

    let common_paths = [
        "/usr/local/cuda/bin/nvcc",
        "/usr/local/cuda-12.6/bin/nvcc",
        "/usr/local/cuda-12/bin/nvcc",
        "/opt/cuda/bin/nvcc",
    ];

    for path in &common_paths {
        if PathBuf::from(path).exists() {
            return Some(path.to_string());
        }
    }

    if Command::new("nvcc").arg("--version").output().is_ok() {
        return Some("nvcc".to_string());
    }

    None
}
