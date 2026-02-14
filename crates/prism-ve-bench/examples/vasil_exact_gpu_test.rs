//! VASIL Exact Metric GPU Test - PATH B COMPLETE VALIDATION
//! NO HALF MEASURES - Full end-to-end GPU pipeline test

use anyhow::{Result, Context};
use chrono::NaiveDate;
use std::collections::HashMap;
use cudarc::driver::CudaContext;
use prism_ve_bench::vasil_exact_metric::{VasilMetricComputer, build_immunity_landscapes};
use prism_ve_bench::data_loader::AllCountriesData;

fn main() -> Result<()> {
    println!("================================================================================");
    println!("🚀 VASIL EXACT METRIC GPU TEST - PATH B VALIDATION");
    println!("================================================================================");
    println!();
    println!("Testing GPU kernel implementation:");
    println!("  1. ✅ compute_weighted_avg_susceptibility (NEW - replaces population×0.5)");
    println!("  2. ✅ compute_gamma_envelopes_batch");
    println!("  3. ✅ classify_gamma_envelopes_batch");
    println!();
    println!("Expected outcome: 77-82% accuracy (was 51.9% with placeholder)");
    println!("================================================================================");
    println!();

    // Population data
    let mut population_map = HashMap::new();
    population_map.insert("Germany".to_string(), 83_200_000.0);
    population_map.insert("USA".to_string(), 331_900_000.0);
    population_map.insert("UK".to_string(), 67_300_000.0);
    population_map.insert("Japan".to_string(), 125_700_000.0);
    population_map.insert("Brazil".to_string(), 214_300_000.0);
    population_map.insert("France".to_string(), 67_400_000.0);
    population_map.insert("Canada".to_string(), 38_200_000.0);
    population_map.insert("Denmark".to_string(), 5_800_000.0);
    population_map.insert("Australia".to_string(), 25_700_000.0);
    population_map.insert("Sweden".to_string(), 10_400_000.0);
    population_map.insert("Mexico".to_string(), 128_000_000.0);
    population_map.insert("SouthAfrica".to_string(), 60_000_000.0);

    // VASIL reference accuracies
    let mut vasil_ref = HashMap::new();
    vasil_ref.insert("Germany", 0.94_f32);
    vasil_ref.insert("USA", 0.91_f32);
    vasil_ref.insert("UK", 0.93_f32);
    vasil_ref.insert("Japan", 0.90_f32);
    vasil_ref.insert("Brazil", 0.89_f32);
    vasil_ref.insert("France", 0.92_f32);
    vasil_ref.insert("Canada", 0.91_f32);
    vasil_ref.insert("Denmark", 0.93_f32);
    vasil_ref.insert("Australia", 0.90_f32);
    vasil_ref.insert("Sweden", 0.92_f32);
    vasil_ref.insert("Mexico", 0.88_f32);
    vasil_ref.insert("SouthAfrica", 0.87_f32);

    println!("[1/5] Loading VASIL data...");
    let vasil_dir = std::path::PathBuf::from("data/VASIL");
    let all_data = AllCountriesData::load_all_vasil_countries(&vasil_dir)?;
    println!("  ✅ Loaded {} countries", all_data.countries.len());
    println!();

    println!("[2/5] Building immunity landscapes...");
    let landscapes = build_immunity_landscapes(&all_data.countries, &population_map);
    println!("  ✅ Built {} landscapes", landscapes.len());
    println!();

    println!("[3/5] Initializing VASIL metric...");
    let mut vasil_metric = VasilMetricComputer::new();
    vasil_metric.initialize(&all_data.countries[0].dms_data, landscapes);
    println!("  ✅ Initialized");
    println!();

    println!("[4/5] Building GPU immunity cache...");
    println!("  GPU Kernels:");
    println!("    - build_p_neut (75 PK combinations)");
    println!("    - compute_immunity (75 PK matrices)");
    println!("    - compute_weighted_avg_susceptibility (NEW!)");
    println!("    - compute_gamma_envelopes_batch");
    println!("    - classify_gamma_envelopes_batch");
    println!();
    
    let eval_start = NaiveDate::from_ymd_opt(2022, 10, 1).unwrap();
    let eval_end = NaiveDate::from_ymd_opt(2023, 10, 31).unwrap();
    
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    
    let t0 = std::time::Instant::now();
    vasil_metric.build_immunity_cache(
        &all_data.countries[0].dms_data,
        &all_data.countries,
        eval_start,
        eval_end,
        &ctx,
        &stream,
    );
    println!("  ✅ GPU cache built in {:.1}s", t0.elapsed().as_secs_f32());
    println!();

    println!("[5/5] Running VASIL exact metric...");
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
    println!("📊 RESULTS");
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
    println!("🎯 VERDICT");
    println!("================================================================================");
    if mean >= 0.77 {
        println!("✅ SUCCESS - PATH B BASELINE ACHIEVED!");
        println!("   Accuracy: {:.1}% (target: 77-82%)", mean * 100.0);
        println!("   GPU weighted_avg kernel: WORKING ✅");
    } else if mean >= 0.65 {
        println!("⚠️  PARTIAL - Close but needs investigation");
        println!("   Accuracy: {:.1}% (target: 77-82%)", mean * 100.0);
    } else {
        println!("❌ FAILURE - GPU kernel not working");
        println!("   Accuracy: {:.1}% (target: 77-82%)", mean * 100.0);
    }
    println!("================================================================================");
    
    Ok(())
}
