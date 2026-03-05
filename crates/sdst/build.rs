//! Build script for the sdst Rust crate.
//!
//! Links against the pre-built libsdst.so (crates/sdst/lib/) and embeds an
//! RPATH so the nhs_rt_full binary finds the library at runtime without
//! requiring LD_LIBRARY_PATH to be set explicitly.

use std::env;

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR must be set by cargo");
    let lib_dir = format!("{}/lib", manifest_dir);

    // Tell the linker where to find libsdst.so at compile time
    println!("cargo:rustc-link-search=native={}", lib_dir);
    println!("cargo:rustc-link-lib=dylib=sdst");

    // cudart is always available when building with the gpu feature on this machine
    println!("cargo:rustc-link-lib=dylib=cudart");

    // Embed RPATH so the binary finds libsdst.so at runtime (avoids needing LD_LIBRARY_PATH)
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir);

    // Re-link if the library changes
    println!("cargo:rerun-if-changed=lib/libsdst.so");
    println!("cargo:rerun-if-changed=include/sdst_api.h");
}
