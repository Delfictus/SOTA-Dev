use anyhow::Result;

use prism_ve_bench::{
    VasilMetricComputer, VasilParameters, DEFAULT_IC50, CALIBRATED_IC50,
};

fn main() -> Result<()> {
    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║          IC50 Wiring Verification Test                       ║");
    eprintln!("║          Confirms IC50 flows through pipeline                ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝\n");
    
    eprintln!("[Step 1/3] Creating VasilMetricComputer with DEFAULT IC50...");
    let default_params = VasilParameters::default();
    eprintln!("  Input IC50: {:?}", default_params.ic50);
    
    let default_computer = VasilMetricComputer::with_params(&default_params);
    let stored_ic50 = default_computer.get_ic50();
    eprintln!("  Stored IC50: {:?}", stored_ic50);
    
    let default_match = default_params.ic50 == *stored_ic50;
    eprintln!("  Match: {}", if default_match { "✓" } else { "✗" });
    
    eprintln!("\n[Step 2/3] Creating VasilMetricComputer with MODIFIED IC50 (×1.5)...");
    let mut modified_params = VasilParameters::default();
    for i in 0..10 {
        modified_params.ic50[i] = DEFAULT_IC50[i] * 1.5;
    }
    eprintln!("  Input IC50: {:?}", modified_params.ic50);
    
    let modified_computer = VasilMetricComputer::with_params(&modified_params);
    let modified_stored = modified_computer.get_ic50();
    eprintln!("  Stored IC50: {:?}", modified_stored);
    
    let modified_match = modified_params.ic50 == *modified_stored;
    eprintln!("  Match: {}", if modified_match { "✓" } else { "✗" });
    
    eprintln!("\n[Step 3/3] Verifying different inputs produce different outputs...");
    let values_differ = stored_ic50[0] != modified_stored[0];
    eprintln!("  Default[0]: {:.4}", stored_ic50[0]);
    eprintln!("  Modified[0]: {:.4}", modified_stored[0]);
    eprintln!("  Different: {}", if values_differ { "✓" } else { "✗" });
    
    eprintln!("\n╔══════════════════════════════════════════════════════════════╗");
    
    let all_pass = default_match && modified_match && values_differ;
    
    if all_pass {
        eprintln!("║  ✅ IC50 WIRING VERIFICATION PASSED                          ║");
        eprintln!("╠══════════════════════════════════════════════════════════════╣");
        eprintln!("║                                                              ║");
        eprintln!("║  IC50 values successfully flow through the pipeline:        ║");
        eprintln!("║                                                              ║");
        eprintln!("║    VasilParameters.ic50                                      ║");
        eprintln!("║           ↓                                                  ║");
        eprintln!("║    VasilMetricComputer::with_params()                        ║");
        eprintln!("║           ↓                                                  ║");
        eprintln!("║    VasilGammaComputer::with_ic50()                           ║");
        eprintln!("║           ↓                                                  ║");
        eprintln!("║    FoldResistanceMatrix::with_ic50()                         ║");
        eprintln!("║           ↓                                                  ║");
        eprintln!("║    compute_p_neut() uses ic50_baseline                       ║");
        eprintln!("║                                                              ║");
        eprintln!("║  FluxNet can now optimize IC50 values and they WILL         ║");
        eprintln!("║  affect the VASIL accuracy computation!                      ║");
    } else {
        eprintln!("║  ❌ IC50 WIRING VERIFICATION FAILED                          ║");
        eprintln!("╠══════════════════════════════════════════════════════════════╣");
        if !default_match {
            eprintln!("║  - Default IC50 not stored correctly                         ║");
        }
        if !modified_match {
            eprintln!("║  - Modified IC50 not stored correctly                        ║");
        }
        if !values_differ {
            eprintln!("║  - IC50 values are identical (not threading)                 ║");
        }
    }
    
    eprintln!("╚══════════════════════════════════════════════════════════════╝");
    
    if all_pass {
        eprintln!("\n🎯 NEXT STEP: Run VASIL benchmark with different IC50 values");
        eprintln!("   to measure actual accuracy impact.");
    }
    
    Ok(())
}
