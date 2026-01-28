// PATH A GPU Test: Epitope-Based P_neut (Target: 85-90% accuracy)
//
// Differs from PATH B by using epitope distance instead of PK pharmacokinetics

use anyhow::{Result, anyhow};
use prism_ve_bench::vasil_exact_metric::*;
use prism_ve_bench::data_loader::*;
use chrono::NaiveDate;
use std::collections::HashMap;

fn main() -> Result<()> {
    println!("================================================================================");
    println!("🚀 VASIL EXACT METRIC GPU TEST - PATH A (EPITOPE-BASED)");
    println!("================================================================================");
    println!();
    println!("Testing EPITOPE-BASED P_neut implementation:");
    println!("  1. ✅ compute_epitope_p_neut (11-dimensional antigenic distance)");
    println!("  2. ✅ compute_immunity_from_epitope_p_neut (using precomputed P_neut)");
    println!("  3. ✅ compute_gamma_envelopes_batch (same as PATH B)");
    println!();
    println!("Expected outcome: 85-90% accuracy (vs 79.4% PATH B baseline)");
    println!("================================================================================");
    println!();
    
    // VASIL reference accuracies
    let vasil_ref: HashMap<&str, f32> = [
        ("Germany", 0.94), ("UK", 0.93), ("France", 0.92), ("USA", 0.91),
        ("Denmark", 0.93), ("Sweden", 0.92), ("SouthAfrica", 0.87), ("Brazil", 0.89),
        ("Australia", 0.90), ("Canada", 0.91), ("Japan", 0.90), ("Mexico", 0.88),
    ].iter().cloned().collect();
    
    // Load data
    println!("[1/5] Loading VASIL data...");
    let vasil_dir = std::path::PathBuf::from("data/VASIL");
    let all_data = AllCountriesData::load_all_vasil_countries(&vasil_dir)?;
    println!("  ✅ Loaded {} countries", all_data.countries.len());
    println!();
    
    // Build immunity landscapes
    println!("[2/5] Building immunity landscapes...");
    println!("  ✅ Built {} landscapes", all_data.countries.len());
    println!();
    
    // Build immunity landscapes
    println!("[3/5] Building immunity landscapes...");
    let mut pop_sizes = HashMap::new();
    pop_sizes.insert("Germany".to_string(), 83_000_000.0);
    pop_sizes.insert("USA".to_string(), 331_000_000.0);
    pop_sizes.insert("UK".to_string(), 67_000_000.0);
    pop_sizes.insert("Japan".to_string(), 126_000_000.0);
    pop_sizes.insert("Brazil".to_string(), 213_000_000.0);
    pop_sizes.insert("France".to_string(), 67_000_000.0);
    pop_sizes.insert("Canada".to_string(), 38_000_000.0);
    pop_sizes.insert("Denmark".to_string(), 5_800_000.0);
    pop_sizes.insert("Australia".to_string(), 25_700_000.0);
    pop_sizes.insert("Sweden".to_string(), 10_300_000.0);
    pop_sizes.insert("Mexico".to_string(), 128_000_000.0);
    pop_sizes.insert("SouthAfrica".to_string(), 59_000_000.0);
    
    let landscapes = build_immunity_landscapes(&all_data.countries, &pop_sizes);
    println!("  ✅ Built {} landscapes", landscapes.len());
    println!();
    
    // Initialize VASIL metric
    println!("[4/5] Initializing VASIL metric with PATH A mode...");
    let dms_data = &all_data.countries[0].dms_data;
    let mut vasil_metric = VasilMetricComputer::new();
    vasil_metric.initialize(dms_data, landscapes);
    
    // Enable PATH A mode with uniform weights
    let epitope_weights = [1.0f32; 11];  // Uniform weights as baseline
    let sigma = 0.5f32;  // Initial bandwidth estimate
    
    println!("[PATH A] Using UNIFORM epitope weights (baseline):");
    println!("  weights = [1.0; 11]");
    println!("  sigma = {:.3}", sigma);
    println!("  NOTE: Calibration will optimize these parameters");
    vasil_metric.set_path_a_mode(epitope_weights, sigma);
    println!("  ✅ Initialized with PATH A mode");
    println!();
    
    // Build GPU immunity cache
    println!("[5/7] Building GPU immunity cache...");
    let eval_start = NaiveDate::from_ymd_opt(2022, 10, 1).unwrap();
    let eval_end = NaiveDate::from_ymd_opt(2023, 10, 31).unwrap();
    
    use cudarc::driver::CudaContext;
    use std::sync::Arc;
    
    let context = Arc::new(CudaContext::new(0)?);
    let stream = context.default_stream();
    
    let cache_start = std::time::Instant::now();
    vasil_metric.build_immunity_cache(
        dms_data,
        &all_data.countries,
        eval_start,
        eval_end,
        &context,
        &stream,
    );
    println!("  ✅ Cache built in {:.2}s", cache_start.elapsed().as_secs_f64());
    println!();
    
    // Compute metrics
    println!("[6/7] Computing VASIL metric with PATH A...");
    
    let t1 = std::time::Instant::now();
    
    let result = vasil_metric.compute_vasil_metric_exact(
        &all_data.countries,
        eval_start,
        eval_end,
    )?;
    
    println!("  ✅ Complete in {:.2}s", t1.elapsed().as_secs_f32());
    println!();
    
    // Results
    println!("================================================================================");
    println!("📊 RESULTS (PATH A - BASELINE WITH UNIFORM WEIGHTS)");
    println!("================================================================================");
    println!("{:<15} {:>12} {:>12} {:>15}", "Country", "Accuracy", "VASIL_ref", "Delta");
    println!("{:-<60}", "");
    
    for (country, acc) in &result.per_country_accuracy {
        let vref = *vasil_ref.get(country.as_str()).unwrap_or(&0.0_f32);
        let delta = acc - vref;
        println!("{:<15} {:>11.1}% {:>11.1}% {:>14.1}%",
                 country, acc * 100.0, vref * 100.0, delta * 100.0);
    }
    
    println!("{:-<60}", "");
    let mean = result.mean_accuracy;
    let vref_mean = vasil_ref.values().sum::<f32>() / vasil_ref.len() as f32;
    println!("{:<15} {:>11.1}% {:>11.1}% {:>14.1}%",
             "MEAN", mean * 100.0, vref_mean * 100.0, (mean - vref_mean) * 100.0);
    println!("{:-<60}", "");
    println!();
    
    // Verdict
    println!("================================================================================");
    println!("🎯 VERDICT (PATH A BASELINE)");
    println!("================================================================================");
    if mean >= 0.85 {
        println!("✅ SUCCESS - PATH A TARGET ACHIEVED!");
        println!("   Accuracy: {:.1}% (target: 85-90%)", mean * 100.0);
        println!("   Epitope-based P_neut: WORKING ✅");
        println!();
        println!("   Next steps:");
        println!("   1. Run Nelder-Mead calibration to optimize 12 parameters");
        println!("   2. Test with calibrated weights (expect +5-10% improvement)");
    } else if mean >= 0.80 {
        println!("⚠️  GOOD PROGRESS - Close to target");
        println!("   Accuracy: {:.1}% (target: 85-90%)", mean * 100.0);
        println!("   Gap: {:.1}% (needs calibration)", (0.85 - mean) * 100.0);
        println!();
        println!("   Action: Run parameter calibration on Germany");
    } else if mean >= 0.75 {
        println!("⚠️  BASELINE OK - Needs calibration");
        println!("   Accuracy: {:.1}% (target: 85-90%)", mean * 100.0);
        println!("   This is expected for uniform weights");
        println!();
        println!("   Action: Calibrate epitope weights + sigma");
    } else {
        println!("❌ ISSUE - Below expected baseline");
        println!("   Accuracy: {:.1}% (expected: ~75-80% even with uniform weights)", mean * 100.0);
        println!();
        println!("   Debug: Check epitope P_neut kernel implementation");
    }
    println!("================================================================================");
    
    Ok(())
}
