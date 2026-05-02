//! Build script for prism-nhs
//!
//! Compiles CUDA kernels to PTX when the `gpu` feature is enabled.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/cuda/spike_density.cu");
    println!("cargo:rerun-if-changed=src/cuda/spike_to_cluster_4d.cu");
    println!("cargo:rerun-if-changed=src/cuda/spike_to_cluster_4d.cuh");
    println!("cargo:rerun-if-changed=src/cuda/lbvh_morton.cu");
    println!("cargo:rerun-if-changed=src/cuda/lbvh_morton.cuh");
    println!("cargo:rerun-if-changed=src/cuda/vram_pool.cu");
    println!("cargo:rerun-if-changed=src/cuda/vram_pool.cuh");
    println!("cargo:rerun-if-changed=src/cuda/gpu_invariant.cu");
    println!("cargo:rerun-if-changed=src/cuda/gpu_invariant.cuh");
    println!("cargo:rerun-if-changed=src/cuda/pre_rank.cu");
    println!("cargo:rerun-if-changed=src/cuda/pre_rank.cuh");
    println!("cargo:rerun-if-changed=src/cuda/rich_spike.cu");
    println!("cargo:rerun-if-changed=src/cuda/rich_spike.cuh");
    println!("cargo:rerun-if-changed=src/cuda/lbvh_tree.cu");
    println!("cargo:rerun-if-changed=src/cuda/lbvh_tree.cuh");
    println!("cargo:rerun-if-changed=src/cuda/sh_basis.cu");
    println!("cargo:rerun-if-changed=src/cuda/sh_basis.cuh");
    println!("cargo:rerun-if-changed=src/cuda/so3_project.cu");
    println!("cargo:rerun-if-changed=src/cuda/so3_project.cuh");
    println!("cargo:rerun-if-changed=src/cuda/adjudicator.cu");
    println!("cargo:rerun-if-changed=src/cuda/adjudicator.cuh");
    println!("cargo:rerun-if-changed=src/cuda/zstr_kernels.cu");
    println!("cargo:rerun-if-changed=src/cuda/zstr_kernels.cuh");
    println!("cargo:rerun-if-changed=src/cuda/asc_steering.cu");
    println!("cargo:rerun-if-changed=src/cuda/asc_steering.cuh");
    println!("cargo:rerun-if-changed=src/cuda/symmetry_consensus.cu");
    println!("cargo:rerun-if-changed=src/cuda/symmetry_consensus.cuh");
    println!("cargo:rerun-if-changed=src/cuda/dynamic_t7.cu");
    println!("cargo:rerun-if-changed=src/cuda/dynamic_t7.cuh");
    println!("cargo:rerun-if-changed=src/cuda/gearbox.cu");
    println!("cargo:rerun-if-changed=src/cuda/gearbox.cuh");

    // Embed RPATH for libsdst.so so the nhs_rt_full binary finds it at runtime.
    // cargo:rustc-link-arg from a dependency build.rs does not propagate to the
    // final binary, so we set it here in the binary crate's build.rs.
    let workspace = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .parent().unwrap()  // crates/
        .parent().unwrap()  // workspace root
        .to_path_buf();
    let sdst_lib = workspace.join("crates/sdst/lib");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", sdst_lib.display());

    let gpu_enabled = env::var("CARGO_FEATURE_GPU").is_ok();
    if !gpu_enabled {
        return;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let nvcc = find_nvcc().expect(
        "nvcc not found. CUDA toolkit is required for the `gpu` feature.\n\
         Set CUDA_HOME or ensure nvcc is in PATH.",
    );

    // M1.2 §3.2 HALT enforcement: run the atomic-or geometric-tie-
    // breaker grep gate against the M1 lane's CUDA sources before
    // invoking nvcc. If the gate fails, abort the build with the
    // gate's own diagnostic — do not proceed to compile a kernel
    // that violates the §M2 anti-legacy-centroid rule.
    run_atomic_or_grep_gate();

    compile_kernel(&nvcc, "src/cuda/spike_density.cu", &out_dir.join("spike_density.ptx"));

    // M1.2: compile the SpikeToCluster4D producer to a static archive
    // (NOT PTX). This .cu file contains both __global__ kernels and
    // extern "C" host-side orchestration functions that call CUB
    // device-wide algorithms (cub::DeviceReduce::Sum,
    // cub::DeviceSegmentedReduce, cub::DeviceRadixSort::SortPairs).
    // CUB device-wide algorithms internally launch kernels from host
    // code, so a PTX target won't link them — we need a static archive
    // that the prism-nhs lib can link against directly. cudart and
    // libstdc++ are pulled in transitively because the .a contains
    // cudaMemcpyAsync / cudaStreamSynchronize calls.
    compile_to_static_archive(
        &nvcc,
        "src/cuda/spike_to_cluster_4d.cu",
        "spike_to_cluster_4d",
        &out_dir,
    );

    // LBVH lane Phase 1: Morton 30-bit encoder. Same static-archive
    // path as the M1 producer — the .cu defines a __global__ kernel
    // launched from an extern "C" host orchestrator. No CUB usage in
    // this archive (Phase 1 is a single kernel). Phase 2 (Karras
    // tree builder + sort) will land alongside.
    compile_to_static_archive(
        &nvcc,
        "src/cuda/lbvh_morton.cu",
        "lbvh_morton",
        &out_dir,
    );

    // F2 lane: stream-ordered memory pool (cudaMemPool_t-backed) +
    // VRAM audit telemetry struct. Three single-thread atomic
    // kernels (init / record_alloc / record_free) plus four host
    // orchestrators that wrap cudaMemPool runtime API calls. Static
    // archive so the cudaMemPool_t / cudaMallocFromPoolAsync runtime
    // calls link in.
    compile_to_static_archive(
        &nvcc,
        "src/cuda/vram_pool.cu",
        "vram_pool",
        &out_dir,
    );

    // Rectification Phase 1: hard-trap GPU invariant enforcement.
    // Single audit kernel + gpu_hard_assert __device__ helper that
    // fires the PTX `trap` instruction on invariant violation. The
    // M1 Conservation-of-Mass audit is the first consumer; future
    // M1 / M2 invariants follow the same pattern.
    compile_to_static_archive(
        &nvcc,
        "src/cuda/gpu_invariant.cu",
        "gpu_invariant",
        &out_dir,
    );

    // Continuous Learning Architecture Phase 1: RichSpike 64-byte
    // schema. Header-only struct + helpers; .cu is the link-probe
    // anchor. Future kernels (Morton encoder upgrade, telemetry
    // compression) extend this archive.
    compile_to_static_archive(
        &nvcc,
        "src/cuda/rich_spike.cu",
        "rich_spike",
        &out_dir,
    );

    // LBVH Phase 2: Karras 2012 radix-tree builder + last-arrival
    // atomic-flag bottom-up AABB reduce. Two kernels + one init
    // kernel + extern "C" host orchestration. 64-byte cache-line-
    // aligned LBVHNode for line-aligned LDG.E.128 access during the
    // reduce.
    compile_to_static_archive(
        &nvcc,
        "src/cuda/lbvh_tree.cu",
        "lbvh_tree",
        &out_dir,
    );

    // Rectification Phase 2: shift-left MAR pre-rank adjudicator.
    // Three single-purpose kernels (compute_aabb_volumes,
    // compute_energy_density, pre_rank_adjudicator) plus host
    // orchestrators. The adjudicator's 3-way output is the
    // cudaGraphConditionalNode SWITCH selector. Includes the §2.3
    // SAD-PATH guard: NaN/Inf observables route to Case 2
    // (Violation).
    compile_to_static_archive(
        &nvcc,
        "src/cuda/pre_rank.cu",
        "pre_rank",
        &out_dir,
    );

    // RECT-3.1.a: Spherical Harmonics Y_lm evaluator (Lmax=5).
    // Straight-line PTX device helper + init kernel that populates
    // the device-side K_LM normalization table. Feeds RECT-3.1.b
    // (so3_project_manifold_kernel) which accumulates a_lm via WMMA
    // Tensor Core fragments.
    compile_to_static_archive(
        &nvcc,
        "src/cuda/sh_basis.cu",
        "sh_basis",
        &out_dir,
    );

    // RECT-3.1.b: SO(3) projection kernel + ContactShellTile
    // (alignas(128), 384 B). Consumes RichSpike clusters, evaluates
    // Y_lm per spike via the inline helper from sh_basis.cuh,
    // accumulates per-cluster a_lm via warp-shuffle reduce, computes
    // C_l = Σ_m |a_lm|², writes one ContactShellTile per cluster.
    // RECT-3.1.c will replace the warp-shuffle reduction with WMMA
    // Tensor Core fragments without changing the on-tile layout.
    compile_to_static_archive(
        &nvcc,
        "src/cuda/so3_project.cu",
        "so3_project",
        &out_dir,
    );

    // T0/T1/T2/T3/T4: InterferometricAdjudicator + Quantum-Photonic
    // Bridge + KL-Divergence + ASC + clock64 telemetry (Blackwell
    // Convergence directive). Single 128-byte FFI struct, __constant__
    // μ_01² LUT derived from `crate::config` extinction coefficients,
    // PTX NaN/le0 guards make the kernel a Total Function over all
    // 32-bit f32 inputs. ASC kernel atomicAdds into the existing
    // `fused_engine.rs::d_forces` buffer (Anti-Greenfield § 2.1).
    compile_to_static_archive(
        &nvcc,
        "src/cuda/adjudicator.cu",
        "adjudicator",
        &out_dir,
    );

    // ZSTR Phase 1: Zero-Stall Telemetry Ring kernels.
    // zstr_signal_completion_kernel: __threadfence_system() + pinned fence
    // write at end of captured graph epoch.
    // zstr_pos_stage_f4_kernel: vectorized LDG.E.128 → STG.E.128 position
    // staging into pinned triple-buffer slot.
    compile_to_static_archive(
        &nvcc,
        "src/cuda/zstr_kernels.cu",
        "zstr_kernels",
        &out_dir,
    );

    // ZSTR Phase 2: Vectorized ASC repulsion force injection.
    // asc_inject_repulsion_v4_kernel: float4 loads + atom.global.add.v4.f32
    // for concurrent 4-wide force accumulation into d_forces.
    // Gate G22_ATOMIC_v4_VERIFIED: ptxas audit required post-compile.
    compile_to_static_archive(
        &nvcc,
        "src/cuda/asc_steering.cu",
        "asc_steering",
        &out_dir,
    );

    // G28 SISR (Spatially-Indexed Symmetric Reflection) — bilateral-truth
    // symmetry consensus gate for homodimer targets.  Operates on
    // ContactShellTile array between SO(3) projection and Adjudicator
    // step inside the captured graph window.  Brute-force pairwise
    // centroid-distance check (n_clusters ≤ 64); writes a u64 prune
    // mask consumed by the Adjudicator step kernel.
    compile_to_static_archive(
        &nvcc,
        "src/cuda/symmetry_consensus.cu",
        "symmetry_consensus",
        &out_dir,
    );

    // Wave 3 / Path B — Dynamic T7 substrate-aware noise-floor calibration.
    // Three captured kernels (capture / reduce / apply) run inside the V2
    // pipeline graph downstream of the Adjudicator step.  All GPU-native;
    // no host bridges per launch.  Replaces the locked 4LPK priors with
    // values derived from the running substrate's thermal-equilibrium Δ_AB
    // distribution after PRISM_DYNT7_N_MIN samples accumulate.
    compile_to_static_archive(
        &nvcc,
        "src/cuda/dynamic_t7.cu",
        "dynamic_t7",
        &out_dir,
    );

    // Wave B.1 — G26 Chronometric Gearbox foundation.  __constant__ float
    // d_gearbox_table[16] (3 active gears + NaN-sentinel Gear 3) +
    // ChronometricStateTensor (16-byte F2-pool persistent state) +
    // prism_gearbox_pointer_swap_kernel (Stateful Finite Automaton: reads
    // adj->adjudication_code, mutates cruise tensor, writes the active
    // gear's dt through *(adj->d_dt) — the address T12 Pre-Flight wired
    // to &d_protocol->dt).  Hello-world test in gearbox.rs proves the
    // 2.0fs → 4.0fs flip without host intervention.
    compile_to_static_archive(
        &nvcc,
        "src/cuda/gearbox.cu",
        "gearbox",
        &out_dir,
    );
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

/// Run the M1 lane's atomic-or geometric-tie-breaker grep gate
/// (`scripts/m1_atomic_or_grep.sh`) against the M1 CUDA sources.
/// If the gate exits non-zero, panic the build script — refusing to
/// produce a kernel that violates blueprint §M2 / M1 contract §3.2.
///
/// The gate's own diagnostics are written to stderr; this helper
/// only reports the script's pass/fail outcome and re-emits the
/// summary line into the cargo log.
fn run_atomic_or_grep_gate() {
    let workspace = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .parent().unwrap()
        .parent().unwrap()
        .to_path_buf();
    let script = workspace.join("scripts/m1_atomic_or_grep.sh");
    if !script.exists() {
        // Be tolerant: if the script is intentionally absent (e.g.
        // mid-rebase), do not block the build. The gate is a
        // belt-and-suspenders check; the unit test in
        // spike_to_cluster_4d.rs::tests is the primary enforcement.
        println!(
            "cargo:warning=M1 atomic-or grep gate script missing: {}",
            script.display()
        );
        return;
    }
    println!(
        "cargo:rerun-if-changed={}",
        script.display()
    );

    let status = Command::new(&script)
        .current_dir(&workspace)
        .status()
        .expect("Failed to invoke atomic-or grep gate script");

    if !status.success() {
        panic!(
            "M1 atomic-or geometric-tie-breaker grep gate FAILED. \
             See stderr above for offending lines. Per blueprint §M2 / \
             M1 contract §3.2, this is a LANE_BLOCKED-class violation \
             at build time."
        );
    }
}

/// Compile a `.cu` file to a host+device object file, then archive it
/// into a static library, and emit the link directives so the host
/// orchestration entry points (e.g. `prism_m1_spike_to_cluster_4d_run`)
/// resolve from Rust extern "C" call sites.
///
/// Used for `.cu` files that contain CUB device-wide algorithm calls
/// (which internally launch their own kernels from host code) — these
/// can NOT be compiled to PTX because PTX is device-only.
fn compile_to_static_archive(
    nvcc: &str,
    source: &str,
    lib_name: &str,
    out_dir: &PathBuf,
) {
    let obj_path = out_dir.join(format!("{}.o", lib_name));
    let lib_path = out_dir.join(format!("lib{}.a", lib_name));

    println!(
        "cargo:info=Compiling {} -> {} (object)",
        source,
        obj_path.display()
    );

    let nvcc_status = Command::new(nvcc)
        .arg("-c")
        .arg("--compile")
        .arg("-Xcompiler").arg("-fPIC")
        .arg("-Xcompiler").arg("-Wall")
        .arg("-O3")
        .arg("--use_fast_math")
        .arg("--restrict")
        .arg("-arch=sm_120")
        .arg("-std=c++17")
        .arg("--expt-relaxed-constexpr")
        .arg("-I/usr/local/cuda/include")
        // CUDA 13+ ships CUB and Thrust under the CCCL umbrella at
        // include/cccl/{cub,thrust,cuda}/. Earlier toolkits had them
        // directly under include/. The extra include path lets
        // `#include <cub/cub.cuh>` resolve under either layout.
        .arg("-I/usr/local/cuda/include/cccl")
        .arg("-Isrc/cuda")
        .arg("-o").arg(&obj_path)
        .arg(source)
        .status()
        .expect("Failed to execute nvcc for static-archive compile");

    if !nvcc_status.success() {
        panic!("nvcc compilation failed for {}", source);
    }

    println!(
        "cargo:info=Archiving {} -> {}",
        obj_path.display(),
        lib_path.display()
    );

    // Remove a stale archive before re-creating, otherwise `ar rcs`
    // would append rather than replace the object on incremental rebuilds.
    let _ = std::fs::remove_file(&lib_path);

    let ar_status = Command::new("ar")
        .arg("rcs")
        .arg(&lib_path)
        .arg(&obj_path)
        .status()
        .expect("Failed to execute ar");

    if !ar_status.success() {
        panic!("ar archive failed for {}", obj_path.display());
    }

    println!("cargo:info=Static archive: {}", lib_path.display());
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static={}", lib_name);
    // CUDA runtime + C++ stdlib are pulled in transitively by the
    // archive's cudaStreamSynchronize / cudaMemcpyAsync calls and the
    // CUB code's STL usage.
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
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
