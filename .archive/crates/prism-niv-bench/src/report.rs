use crate::benchmark::BenchmarkResults;
use crate::Result;
use std::fs::File;
use std::io::Write;
use std::path::Path;

pub fn generate_report(results: &BenchmarkResults, output_path: &Path) -> Result<()> {
    let mut file = File::create(output_path)?;
    
    writeln!(file, "# NiV-Bench Results Report")?;
    writeln!(file, "Generated at: {}\n", chrono::Local::now())?;
    
    writeln!(file, "## Quick Summary")?;
    writeln!(file, "- Cryptic AUC: {:.3}", results.cryptic.auc_roc)?;
    writeln!(file, "- Epitope Precision@20: {:.3}", results.epitope.precision_at_20)?;
    writeln!(file, "- DDG Spearman: {:.3}", results.ddg.spearman_rho)?;
    writeln!(file, "- Speed: {:.1} structures/sec (x{:.0} over baseline)", 
             results.speed.structures_per_second, results.speed.speedup_factor)?;
             
    writeln!(file, "\n## Detailed Metrics")?;
    writeln!(file, "### Cryptic Site Detection")?;
    writeln!(file, "AUC-ROC: {:.3}", results.cryptic.auc_roc)?;
    writeln!(file, "Precision: {:.3}", results.cryptic.precision)?;
    writeln!(file, "Recall: {:.3}", results.cryptic.recall)?;
    writeln!(file, "pRMSD: {:.3}", results.cryptic.p_rmsd)?;
    
    writeln!(file, "\n### Epitope Prediction")?;
    writeln!(file, "Precision@10: {:.3}", results.epitope.precision_at_10)?;
    writeln!(file, "Precision@20: {:.3}", results.epitope.precision_at_20)?;
    writeln!(file, "Recall: {:.3}", results.epitope.recall)?;
    
    writeln!(file, "\n### DDG Prediction")?;
    writeln!(file, "Spearman Rho: {:.3}", results.ddg.spearman_rho)?;
    writeln!(file, "RMSE: {:.3}", results.ddg.rmse)?;
    
    // JSON dump for convenience
    let json = serde_json::to_string_pretty(results).unwrap_or_default();
    writeln!(file, "\n## RAW JSON")?;
    writeln!(file, "```json\n{}\n```", json)?;
    
    Ok(())
}