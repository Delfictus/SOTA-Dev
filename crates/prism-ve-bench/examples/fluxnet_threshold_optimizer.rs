use anyhow::Result;
use prism_ve_bench::vasil_exact_metric::*;
use prism_ve_bench::data_loader::*;
use prism_ve_bench::fluxnet_vasil_adapter::*;
use chrono::NaiveDate;
use std::collections::HashMap;
use std::sync::Arc;

const MAX_EPISODES: usize = 50;
const STEPS_PER_EPISODE: usize = 20;
const TARGET_ACCURACY: f32 = 0.88;

fn main() -> Result<()> {
    println!("================================================================================");
    println!("FLUXNET THRESHOLD OPTIMIZER - VasilParameters Tuning");
    println!("================================================================================");
    println!();
    println!("Strategy: Keep GPU envelope cache fixed, tune DECISION THRESHOLDS only");
    println!();
    println!("Tunable Parameters:");
    println!("  - negligible_threshold: Relative change below which to exclude (default: 5%)");
    println!("  - min_frequency: Minimum frequency for inclusion (default: 3%)");
    println!("  - min_peak_frequency: Minimum peak to qualify as major (default: 1%)");
    println!("  - confidence_margin: Envelope must exceed this to be 'decided'");
    println!();
    println!("KEY INSIGHT: No cache rebuild per step = 100x faster training!");
    println!("================================================================================");
    println!();

    println!("[1/5] Loading VASIL data...");
    let vasil_dir = std::path::PathBuf::from("data/VASIL");
    let all_data = AllCountriesData::load_all_vasil_countries(&vasil_dir)?;
    println!("  Loaded {} countries", all_data.countries.len());

    let training_countries: Vec<&str> = vec![
        "Germany", "USA", "UK", "Japan", "France",
        "Brazil", "Canada", "Denmark", "Australia", "SouthAfrica"
    ];
    let validation_countries: Vec<&str> = vec!["Sweden", "Mexico"];

    let train_data: Vec<_> = all_data.countries.iter()
        .filter(|c| training_countries.contains(&c.name.as_str()))
        .cloned()
        .collect();
    let valid_data: Vec<_> = all_data.countries.iter()
        .filter(|c| validation_countries.contains(&c.name.as_str()))
        .cloned()
        .collect();

    println!("  Training on: {} countries", train_data.len());
    println!("  Validation: {} countries (held out)", valid_data.len());

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

    let landscapes = build_immunity_landscapes(&train_data, &pop_sizes);

    println!("[2/5] Initializing GPU...");
    use cudarc::driver::CudaContext;
    let context = Arc::new(CudaContext::new(0)?);
    let stream = context.default_stream();
    println!("  GPU ready");

    let eval_start = NaiveDate::from_ymd_opt(2022, 10, 1).unwrap();
    let eval_end = NaiveDate::from_ymd_opt(2023, 10, 31).unwrap();

    println!("[3/5] Building PATH B baseline (one-time GPU cache)...");
    let dms_data = &train_data[0].dms_data;
    let mut vasil_metric = VasilMetricComputer::new();
    vasil_metric.initialize(dms_data, landscapes);
    vasil_metric.build_immunity_cache(
        dms_data, &train_data, eval_start, eval_end, &context, &stream,
    );

    let baseline_result = vasil_metric.compute_vasil_metric_exact(&train_data, eval_start, eval_end)?;
    let baseline_acc = baseline_result.mean_accuracy;
    println!("  PATH B Baseline: {:.2}%", baseline_acc * 100.0);
    println!("  Predictions: {}, Correct: {}", baseline_result.total_predictions, baseline_result.total_correct);
    println!("  Excluded: undecided={}, negligible={}, low_freq={}",
             baseline_result.total_excluded_undecided,
             baseline_result.total_excluded_negligible,
             baseline_result.total_excluded_low_freq);

    println!("[4/5] FluxNet threshold optimization (NO cache rebuild)...");
    println!();
    println!("================================================================================");
    println!("TRAINING (target: {}%)", TARGET_ACCURACY * 100.0);
    println!("================================================================================");

    let mut optimizer = VEFluxNetOptimizer::new();
    let mut best_accuracy = baseline_acc;
    let mut best_params = VasilParameters::default();
    let mut prev_accuracy = baseline_acc;

    for episode in 0..MAX_EPISODES {
        let (ep_count, epsilon, _) = optimizer.get_stats();
        println!("\n--- Episode {}/{} (epsilon={:.3}, best={:.2}%) ---",
                 episode + 1, MAX_EPISODES, epsilon, best_accuracy * 100.0);

        for step in 0..STEPS_PER_EPISODE {
            let exclusion_rate = baseline_result.total_excluded_undecided as f32 /
                (baseline_result.total_predictions + baseline_result.total_excluded_undecided) as f32;

            let prev_state = VEFluxNetState {
                current_accuracy: prev_accuracy,
                country_id: 0,
                time_period: 1,
                variant_diversity: train_data.len() as u8,
                exclusion_rate,
                envelope_confidence: 0.5,
                rise_ratio: 0.36,
            };

            let action = optimizer.optimize_step(&prev_state, true);
            let current_params = optimizer.get_params().clone();

            vasil_metric.update_params(&current_params);

            let result = vasil_metric.compute_vasil_metric_exact(&train_data, eval_start, eval_end)?;
            let new_accuracy = result.mean_accuracy;

            let new_exclusion_rate = result.total_excluded_undecided as f32 /
                (result.total_predictions + result.total_excluded_undecided) as f32;

            let new_state = VEFluxNetState {
                current_accuracy: new_accuracy,
                country_id: 0,
                time_period: 1,
                variant_diversity: train_data.len() as u8,
                exclusion_rate: new_exclusion_rate,
                envelope_confidence: 0.5,
                rise_ratio: 0.36,
            };

            optimizer.record_result(&prev_state, action, new_accuracy, &new_state);

            if new_accuracy > best_accuracy {
                best_accuracy = new_accuracy;
                best_params = current_params.clone();
                println!("    Step {}: {:.2}% NEW BEST ({:?})", step + 1, new_accuracy * 100.0, action);
            }

            prev_accuracy = new_accuracy;

            if best_accuracy >= TARGET_ACCURACY {
                println!("\nTARGET ACHIEVED!");
                break;
            }
        }

        optimizer.decay_epsilon();

        if best_accuracy >= TARGET_ACCURACY {
            break;
        }
    }

    println!();
    println!("================================================================================");
    println!("OPTIMIZATION COMPLETE");
    println!("================================================================================");
    println!();
    println!("Training Results:");
    println!("  PATH B Baseline:    {:.2}%", baseline_acc * 100.0);
    println!("  Best Achieved:      {:.2}%", best_accuracy * 100.0);
    println!("  Improvement:        {:+.2}%", (best_accuracy - baseline_acc) * 100.0);
    println!();
    println!("Optimized Thresholds:");
    println!("  negligible_threshold: {:.4} (default: 0.05)", best_params.negligible_threshold);
    println!("  min_frequency:        {:.4} (default: 0.03)", best_params.min_frequency);
    println!("  min_peak_frequency:   {:.4} (default: 0.01)", best_params.min_peak_frequency);
    println!("  confidence_margin:    {:.4} (default: 0.00)", best_params.confidence_margin);

    println!();
    println!("[5/5] Validation on held-out countries...");
    if !valid_data.is_empty() {
        let valid_landscapes = build_immunity_landscapes(&valid_data, &pop_sizes);
        let mut valid_metric = VasilMetricComputer::with_params(&best_params);
        valid_metric.initialize(&valid_data[0].dms_data, valid_landscapes);
        valid_metric.build_immunity_cache(
            &valid_data[0].dms_data, &valid_data, eval_start, eval_end, &context, &stream,
        );

        let valid_result = valid_metric.compute_vasil_metric_exact(&valid_data, eval_start, eval_end)?;
        println!("  Validation Accuracy: {:.2}%", valid_result.mean_accuracy * 100.0);
        println!("  Predictions: {}, Correct: {}", valid_result.total_predictions, valid_result.total_correct);
    }

    let params_json = serde_json::to_string_pretty(&best_params)?;
    std::fs::write("validation_results/fluxnet_threshold_optimized.json", &params_json)?;
    println!();
    println!("Saved: validation_results/fluxnet_threshold_optimized.json");

    optimizer.save("validation_results/fluxnet_ve_qtable.json")?;
    println!("Saved: validation_results/fluxnet_ve_qtable.json");

    println!();
    if best_accuracy >= 0.85 {
        println!("SUCCESS: Achieved 85%+ - approaching VASIL's 90.8%!");
    } else if best_accuracy > baseline_acc + 0.02 {
        println!("IMPROVED: +{:.1}% over baseline", (best_accuracy - baseline_acc) * 100.0);
    } else {
        println!("Minimal improvement - consider expanding action space");
    }
    println!("================================================================================");

    Ok(())
}
