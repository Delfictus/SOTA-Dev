use std::process::Command;
use std::env;
use std::path::{Path, PathBuf};
use std::fs;

fn main() {
    println!("cargo:rerun-if-changed=src/kernels/feature_merge.cu");
    println!("cargo:rerun-if-changed=../prism-gpu/src/kernels/cryptic/");
    println!("cargo:rerun-if-changed=../prism-gpu/src/kernels/mega_fused_batch.ptx");
    println!("cargo:rerun-if-changed=build.rs");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let target_ptx_dir = PathBuf::from("target/ptx"); // For easy access in dev
    fs::create_dir_all(&target_ptx_dir).unwrap();

    // 1. Compile feature_merge.cu
    compile_kernel(
        "src/kernels/feature_merge.cu",
        &out_dir.join("feature_merge.ptx"),
    );
    fs::copy(out_dir.join("feature_merge.ptx"), target_ptx_dir.join("feature_merge.ptx")).ok();

    // 2. Compile Cryptic Kernels
    let cryptic_dir = Path::new("../prism-gpu/src/kernels/cryptic");
    let cryptic_kernels = [
        "cryptic_hessian.cu",
        "cryptic_eigenmodes.cu",
        "cryptic_probe_score.cu",
        "cryptic_signal_fusion.cu"
    ];

    for kernel in cryptic_kernels {
        let source = cryptic_dir.join(kernel);
        let output = out_dir.join(kernel.replace(".cu", ".ptx"));
        if source.exists() {
             compile_kernel(source.to_str().unwrap(), &output);
             fs::copy(&output, target_ptx_dir.join(kernel.replace(".cu", ".ptx"))).ok();
        } else {
            println!("cargo:warning=Cryptic kernel source not found: {:?}", source);
        }
    }

    // 3. Copy mega_fused_batch.ptx
    let mega_batch_ptx = Path::new("../prism-gpu/src/kernels/mega_fused_batch.ptx");
    if mega_batch_ptx.exists() {
        fs::copy(mega_batch_ptx, out_dir.join("mega_fused_batch.ptx")).expect("Failed to copy mega_fused_batch.ptx");
        fs::copy(mega_batch_ptx, target_ptx_dir.join("mega_fused_batch.ptx")).ok();
    } else {
        println!("cargo:warning=mega_fused_batch.ptx not found at {:?}", mega_batch_ptx);
    }
}

fn compile_kernel(source: &str, output: &PathBuf) {
    // Check if nvcc is available
    if Command::new("nvcc").arg("--version").output().is_ok() {
        let status = Command::new("nvcc")
            .args(&["-ptx", "-o"])
            .arg(output)
            .arg(source)
            .arg("-arch=sm_75") // Target T4/Ampere common baseline
            .arg("--use_fast_math")
            .arg("-I../prism-gpu/src/kernels") // Include path for potential headers
            .status()
            .expect("Failed to execute nvcc");

        if !status.success() {
            panic!("nvcc failed to compile kernel: {}", source);
        }
        println!("cargo:warning=Compiled {} to PTX", source);
    } else {
        println!("cargo:warning=nvcc not found, skipping CUDA compilation for {}", source);
        // Create dummy file to prevent build failure if CUDA not present (for CI/non-GPU dev)
        if !output.exists() {
            fs::write(output, "dummy ptx").unwrap();
        }
    }
}
