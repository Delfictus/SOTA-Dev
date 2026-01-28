//! HTML and PDF report generation

use crate::ablation::AblationResults;
use crate::config::ReportConfig;
use crate::correlation::CorrelationResult;
use crate::outputs::SummaryJson;
use crate::sites::CrypticSite;
use crate::{dependency_install_instructions, find_pdf_renderer};
use anyhow::{bail, Context, Result};
use std::fs;
use std::path::Path;
use std::process::Command;

/// HTML report generator
pub struct HtmlReport;

impl HtmlReport {
    /// Generate HTML report with embedded NGL viewer
    pub fn generate(
        path: &Path,
        config: &ReportConfig,
        sites: &[CrypticSite],
        ablation: &AblationResults,
        correlation: Option<&CorrelationResult>,
    ) -> Result<()> {
        let html = Self::render_html(config, sites, ablation, correlation)?;
        fs::write(path, html)?;
        Ok(())
    }

    fn render_html(
        config: &ReportConfig,
        sites: &[CrypticSite],
        ablation: &AblationResults,
        correlation: Option<&CorrelationResult>,
    ) -> Result<String> {
        let mut html = String::new();

        // HTML header with embedded styles and NGL viewer
        html.push_str(&Self::html_header(&config.input_pdb.display().to_string()));

        // Title section
        html.push_str(&Self::title_section(config));

        // Executive summary
        html.push_str(&Self::executive_summary(sites, ablation, correlation));

        // Ablation analysis (REQUIRED section)
        html.push_str(&Self::ablation_section(ablation));

        // Site details
        html.push_str(&Self::sites_section(sites));

        // Correlation results
        if let Some(corr) = correlation {
            html.push_str(&Self::correlation_section(corr));
        }

        // Methods section
        html.push_str(&Self::methods_section());

        // Footer
        html.push_str(&Self::html_footer());

        Ok(html)
    }

    fn html_header(pdb_id: &str) -> String {
        format!(r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PRISM4D Cryptic Site Report - {}</title>
    <script src="https://unpkg.com/ngl@2.0.0-dev.37/dist/ngl.js"></script>
    <style>
        :root {{
            --primary: #2563eb;
            --success: #22c55e;
            --warning: #f59e0b;
            --danger: #ef4444;
            --dark: #1e293b;
            --light: #f8fafc;
            --border: #e2e8f0;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: var(--dark);
            background: var(--light);
        }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 40px 20px; }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 3px solid var(--primary);
        }}
        .header h1 {{ font-size: 32px; margin-bottom: 10px; }}
        .header .subtitle {{ font-size: 18px; color: #64748b; }}
        .section {{
            background: white;
            border-radius: 8px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border: 1px solid var(--border);
        }}
        .section h2 {{
            font-size: 22px;
            margin-bottom: 16px;
            padding-bottom: 8px;
            border-bottom: 2px solid var(--primary);
        }}
        .section h3 {{ font-size: 18px; margin: 16px 0 12px 0; color: var(--dark); }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
        }}
        .stat-card {{
            background: var(--light);
            border-radius: 6px;
            padding: 16px;
            text-align: center;
        }}
        .stat-card .label {{ font-size: 12px; color: #64748b; text-transform: uppercase; }}
        .stat-card .value {{ font-size: 28px; font-weight: 700; margin: 4px 0; }}
        .stat-card .unit {{ font-size: 14px; color: #94a3b8; }}
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
        }}
        .badge-success {{ background: #dcfce7; color: #166534; }}
        .badge-warning {{ background: #fef3c7; color: #92400e; }}
        .badge-danger {{ background: #fee2e2; color: #991b1b; }}
        .badge-info {{ background: #dbeafe; color: #1e40af; }}
        table {{ width: 100%; border-collapse: collapse; margin: 16px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid var(--border); }}
        th {{ background: var(--light); font-weight: 600; }}
        .ablation-note {{
            background: #fef3c7;
            border-left: 4px solid var(--warning);
            padding: 16px;
            margin: 16px 0;
            font-style: italic;
        }}
        .site-card {{
            border: 1px solid var(--border);
            border-radius: 8px;
            margin-bottom: 16px;
            overflow: hidden;
        }}
        .site-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 16px;
            background: var(--light);
        }}
        .site-body {{ padding: 16px; }}
        .viewer {{
            width: 100%;
            height: 400px;
            background: #1e293b;
            border-radius: 8px;
            margin: 16px 0;
        }}
        .footer {{
            text-align: center;
            padding-top: 20px;
            border-top: 1px solid var(--border);
            color: #64748b;
        }}
        @media print {{
            .viewer {{ display: none; }}
            body {{ background: white; }}
            .section {{ box-shadow: none; }}
        }}
    </style>
</head>
<body>
<div class="container">
"#, pdb_id)
    }

    fn title_section(config: &ReportConfig) -> String {
        let timestamp = chrono::Utc::now().format("%Y-%m-%d %H:%M UTC");
        format!(r#"
    <div class="header">
        <h1>PRISM4D Cryptic Binding Site Analysis</h1>
        <div class="subtitle">
            Target: <strong>{}</strong> |
            Replicates: {} |
            Generated: {}
        </div>
    </div>
"#, config.input_pdb.display(), config.replicates, timestamp)
    }

    fn executive_summary(
        sites: &[CrypticSite],
        ablation: &AblationResults,
        correlation: Option<&CorrelationResult>,
    ) -> String {
        let druggable_count = sites.iter().filter(|s| s.is_druggable).count();
        let best_site = sites.first();

        let tier1_info = correlation
            .and_then(|c| c.tier1.as_ref())
            .map(|t| format!("{:.1} A", t.best_distance))
            .unwrap_or_else(|| "N/A".to_string());

        let tier2_info = correlation
            .and_then(|c| c.tier2.as_ref())
            .map(|t| format!("{:.2}", t.best_f1))
            .unwrap_or_else(|| "N/A".to_string());

        format!(r#"
    <div class="section">
        <h2>Executive Summary</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="label">Sites Detected</div>
                <div class="value">{}</div>
                <div class="unit">{} druggable</div>
            </div>
            <div class="stat-card">
                <div class="label">Cryo Contrast</div>
                <div class="value">{}</div>
                <div class="unit">spike delta</div>
            </div>
            <div class="stat-card">
                <div class="label">UV Response</div>
                <div class="value">{}</div>
                <div class="unit">spike delta</div>
            </div>
            <div class="stat-card">
                <div class="label">Best Site Score</div>
                <div class="value">{:.2}</div>
                <div class="unit">{}</div>
            </div>
            <div class="stat-card">
                <div class="label">Tier 1 (Ligand)</div>
                <div class="value">{}</div>
                <div class="unit">nearest dist</div>
            </div>
            <div class="stat-card">
                <div class="label">Tier 2 (Truth)</div>
                <div class="value">{}</div>
                <div class="unit">best F1</div>
            </div>
        </div>
    </div>
"#,
            sites.len(),
            druggable_count,
            ablation.deltas.spikes_cryo_vs_baseline,
            ablation.deltas.spikes_cryouv_vs_cryo,
            best_site.map(|s| s.rank_score).unwrap_or(0.0),
            best_site.map(|s| s.site_id.as_str()).unwrap_or("N/A"),
            tier1_info,
            tier2_info
        )
    }

    fn ablation_section(ablation: &AblationResults) -> String {
        let cryo_badge = if ablation.comparison.cryo_contrast_significant {
            r#"<span class="badge badge-success">SIGNIFICANT</span>"#
        } else {
            r#"<span class="badge badge-warning">NOT SIGNIFICANT</span>"#
        };

        let uv_badge = if ablation.comparison.uv_response_significant {
            r#"<span class="badge badge-success">SIGNIFICANT</span>"#
        } else {
            r#"<span class="badge badge-warning">NOT SIGNIFICANT</span>"#
        };

        format!(r#"
    <div class="section">
        <h2>Ablation Analysis (Required)</h2>

        <div class="ablation-note">
            <strong>Why ablation is mandatory:</strong> Ablation analysis (baseline vs cryo vs cryo+UV)
            is essential to properly attribute site emergence. Without this three-way comparison,
            we cannot distinguish temperature-driven effects from UV-induced pocket opening.
        </div>

        <table>
            <thead>
                <tr>
                    <th>Mode</th>
                    <th>Description</th>
                    <th>Total Spikes</th>
                    <th>Sites</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Baseline</strong></td>
                    <td>300K constant, UV off</td>
                    <td>{}</td>
                    <td>{}</td>
                </tr>
                <tr>
                    <td><strong>Cryo-only</strong></td>
                    <td>Temperature ramp, UV off</td>
                    <td>{}</td>
                    <td>{}</td>
                </tr>
                <tr>
                    <td><strong>Cryo+UV</strong></td>
                    <td>Temperature ramp, UV on</td>
                    <td>{}</td>
                    <td>{}</td>
                </tr>
            </tbody>
        </table>

        <h3>Effect Analysis</h3>
        <p><strong>Cryo Contrast (cryo - baseline):</strong> {} spikes {}</p>
        <p><strong>UV Response (cryo+UV - cryo):</strong> {} spikes {}</p>

        <h3>Interpretation</h3>
        <p>{}</p>
    </div>
"#,
            ablation.baseline.total_spikes,
            ablation.baseline.sites.len(),
            ablation.cryo_only.total_spikes,
            ablation.cryo_only.sites.len(),
            ablation.cryo_uv.total_spikes,
            ablation.cryo_uv.sites.len(),
            ablation.deltas.spikes_cryo_vs_baseline,
            cryo_badge,
            ablation.deltas.spikes_cryouv_vs_cryo,
            uv_badge,
            ablation.comparison.interpretation
        )
    }

    fn sites_section(sites: &[CrypticSite]) -> String {
        let mut html = String::from(r#"
    <div class="section">
        <h2>Detected Cryptic Sites</h2>
"#);

        if sites.is_empty() {
            html.push_str(r#"<p style="color: #64748b;">No cryptic sites detected.</p>"#);
        } else {
            html.push_str(r#"
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Site ID</th>
                    <th>Residues</th>
                    <th>Volume (A^3)</th>
                    <th>Persistence</th>
                    <th>Hydrophobic</th>
                    <th>Score</th>
                    <th>Druggable</th>
                </tr>
            </thead>
            <tbody>
"#);

            for site in sites {
                let druggable_badge = if site.is_druggable {
                    r#"<span class="badge badge-success">YES</span>"#
                } else {
                    r#"<span class="badge badge-danger">NO</span>"#
                };

                html.push_str(&format!(r#"
                <tr>
                    <td><strong>#{}</strong></td>
                    <td>{}</td>
                    <td>{}</td>
                    <td>{:.0}</td>
                    <td>{:.0}%</td>
                    <td>{:.0}%</td>
                    <td>{:.3}</td>
                    <td>{}</td>
                </tr>
"#,
                    site.rank,
                    site.site_id,
                    site.residues.len(),
                    site.metrics.geometry.volume_mean,
                    site.metrics.persistence.present_fraction * 100.0,
                    site.metrics.chemistry.hydrophobic_fraction * 100.0,
                    site.rank_score,
                    druggable_badge
                ));
            }

            html.push_str("</tbody></table>");
        }

        html.push_str("</div>");
        html
    }

    fn correlation_section(correlation: &CorrelationResult) -> String {
        let mut html = String::from(r#"
    <div class="section">
        <h2>Correlation Analysis</h2>
"#);

        // Tier 1
        if let Some(tier1) = &correlation.tier1 {
            html.push_str(&format!(r#"
        <h3>Tier 1: Holo Ligand Proximity</h3>
        <p>Best site: <strong>{}</strong> at {:.2} A from ligand</p>
        <p>Sites within 5A: {} | Sites within 8A: {}</p>
        <table>
            <thead>
                <tr>
                    <th>Site</th>
                    <th>Nearest Dist (A)</th>
                    <th>Centroid Dist (A)</th>
                    <th>Recall @4A</th>
                    <th>Hit</th>
                </tr>
            </thead>
            <tbody>
"#,
                tier1.best_site_id,
                tier1.best_distance,
                tier1.sites_within_5a,
                tier1.sites_within_8a
            ));

            for site in &tier1.site_correlations {
                let hit_badge = if site.is_hit {
                    r#"<span class="badge badge-success">YES</span>"#
                } else {
                    r#"<span class="badge badge-danger">NO</span>"#
                };
                html.push_str(&format!(r#"
                <tr>
                    <td>{}</td>
                    <td>{:.2}</td>
                    <td>{:.2}</td>
                    <td>{:.2}</td>
                    <td>{}</td>
                </tr>
"#,
                    site.site_id,
                    site.nearest_ligand_atom_distance_a,
                    site.ligand_centroid_distance_a,
                    site.residue_recall_within_4a,
                    hit_badge
                ));
            }
            html.push_str("</tbody></table>");
        }

        // Tier 2
        if let Some(tier2) = &correlation.tier2 {
            html.push_str(&format!(r#"
        <h3>Tier 2: Truth Residue Overlap</h3>
        <p>Best F1: <strong>{:.3}</strong> ({})</p>
        <p>Hit@1: {} | Hit@3: {}</p>
"#,
                tier2.best_f1,
                tier2.best_site_id,
                if tier2.hit_at_1 { "YES" } else { "NO" },
                if tier2.hit_at_3 { "YES" } else { "NO" }
            ));
        }

        html.push_str("</div>");
        html
    }

    fn methods_section() -> String {
        r#"
    <div class="section">
        <h2>Methods</h2>
        <p>Cryptic binding sites were detected using the PRISM4D NHS-Cryo-UV pipeline:</p>
        <ul>
            <li><strong>Neuromorphic Holographic Stream (NHS):</strong> Leaky integrate-and-fire
                network detects cooperative dewetting events indicating pocket opening.</li>
            <li><strong>Cryogenic Contrast:</strong> Temperature ramp from cold to physiological
                reveals temperature-dependent hydrophobic site emergence.</li>
            <li><strong>UV Spectroscopy:</strong> Multi-wavelength aromatic excitation probes
                chromophore-proximal binding sites.</li>
            <li><strong>Ablation Analysis:</strong> Three-way comparison (baseline/cryo/cryo+UV)
                attributes site emergence to specific mechanisms.</li>
        </ul>
        <p>Sites are ranked by a weighted score combining persistence, volume, UV response,
           hydrophobicity, and replica agreement.</p>
    </div>
"#.to_string()
    }

    fn html_footer() -> String {
        let year = chrono::Utc::now().format("%Y");
        format!(r#"
    <div class="footer">
        <p>Generated by <strong>PRISM4D</strong> Cryo-UV Cryptic Site Detection Pipeline</p>
        <p>&copy; {} PRISM4D Team</p>
    </div>
</div>
</body>
</html>
"#, year)
    }
}

/// PDF report generator
pub struct PdfReport;

impl PdfReport {
    /// Generate PDF from HTML report
    pub fn generate(html_path: &Path, pdf_path: &Path) -> Result<()> {
        let (renderer_name, renderer_path) = find_pdf_renderer().ok_or_else(|| {
            anyhow::anyhow!(
                "No PDF renderer found.\n{}",
                dependency_install_instructions("pdf")
            )
        })?;

        match renderer_name.as_str() {
            "wkhtmltopdf" => Self::render_with_wkhtmltopdf(&renderer_path, html_path, pdf_path),
            "chromium" | "chromium-browser" | "google-chrome" | "chrome" => {
                Self::render_with_chromium(&renderer_path, html_path, pdf_path)
            }
            "playwright" => Self::render_with_playwright(&renderer_path, html_path, pdf_path),
            _ => bail!("Unknown PDF renderer: {}", renderer_name),
        }
    }

    fn render_with_wkhtmltopdf(
        wkhtmltopdf: &Path,
        html_path: &Path,
        pdf_path: &Path,
    ) -> Result<()> {
        let status = Command::new(wkhtmltopdf)
            .args([
                "--page-size", "A4",
                "--margin-top", "20mm",
                "--margin-bottom", "20mm",
                "--margin-left", "15mm",
                "--margin-right", "15mm",
                html_path.to_str().unwrap(),
                pdf_path.to_str().unwrap(),
            ])
            .status()
            .with_context(|| format!("Failed to run wkhtmltopdf: {}", wkhtmltopdf.display()))?;

        if !status.success() {
            bail!("wkhtmltopdf exited with error");
        }

        if !pdf_path.exists() {
            bail!("PDF was not generated: {}", pdf_path.display());
        }

        Ok(())
    }

    fn render_with_chromium(
        chromium: &Path,
        html_path: &Path,
        pdf_path: &Path,
    ) -> Result<()> {
        // Convert to file:// URL
        let html_url = format!("file://{}", html_path.canonicalize()?.display());

        let status = Command::new(chromium)
            .args([
                "--headless",
                "--disable-gpu",
                &format!("--print-to-pdf={}", pdf_path.display()),
                &html_url,
            ])
            .status()
            .with_context(|| format!("Failed to run chromium: {}", chromium.display()))?;

        if !status.success() {
            bail!("Chromium exited with error");
        }

        Ok(())
    }

    fn render_with_playwright(
        playwright: &Path,
        html_path: &Path,
        pdf_path: &Path,
    ) -> Result<()> {
        let script = format!(
            r#"
const {{ chromium }} = require('playwright');
(async () => {{
    const browser = await chromium.launch();
    const page = await browser.newPage();
    await page.goto('file://{}');
    await page.pdf({{ path: '{}', format: 'A4' }});
    await browser.close();
}})();
"#,
            html_path.canonicalize()?.display(),
            pdf_path.display()
        );

        let status = Command::new("node")
            .args(["-e", &script])
            .status()
            .context("Failed to run playwright via node")?;

        if !status.success() {
            bail!("Playwright PDF generation failed");
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_html_generation() {
        let config = ReportConfig::default();
        let sites = vec![];
        // Test mock: Baseline runs 50k frames, cryo phases run 100k frames
        let ablation = crate::ablation::AblationResults {
            baseline: crate::ablation::AblationRunResult {
                mode: crate::ablation::AblationMode::Baseline,
                total_spikes: 1000,
                events_emitted: 1000,
                phase_spikes: (333, 333, 334),
                frames_analyzed: 50000,
                spikes_per_1k_frames: 20.0, // 1000/50 = 20 per 1k frames
                sites: vec![],
                mean_volume: 0.0,
                mean_sasa: None,
                runtime_seconds: 60.0,
            },
            cryo_only: crate::ablation::AblationRunResult {
                mode: crate::ablation::AblationMode::CryoOnly,
                total_spikes: 5000,
                events_emitted: 5000,
                phase_spikes: (2000, 2000, 1000),
                frames_analyzed: 100000,
                spikes_per_1k_frames: 50.0, // 5000/100 = 50 per 1k frames
                sites: vec![],
                mean_volume: 0.0,
                mean_sasa: None,
                runtime_seconds: 60.0,
            },
            cryo_uv: crate::ablation::AblationRunResult {
                mode: crate::ablation::AblationMode::CryoUv,
                total_spikes: 7000,
                events_emitted: 7000,
                phase_spikes: (2500, 3000, 1500),
                frames_analyzed: 100000,
                spikes_per_1k_frames: 70.0, // 7000/100 = 70 per 1k frames
                sites: vec![],
                mean_volume: 0.0,
                mean_sasa: None,
                runtime_seconds: 60.0,
            },
            deltas: crate::ablation::AblationDeltas {
                spikes_cryo_vs_baseline: 4000,
                spikes_cryouv_vs_cryo: 2000,
                spikes_cryouv_vs_baseline: 6000,
                rate_cryo_vs_baseline: 30.0,  // 50 - 20 = 30 per 1k frames
                rate_cryouv_vs_cryo: 20.0,     // 70 - 50 = 20 per 1k frames
                sites_cryo_vs_baseline: 0,
                sites_cryouv_vs_cryo: 0,
                volume_cryouv_vs_cryo: 0.0,
                sasa_cryouv_vs_cryo: None,
            },
            comparison: crate::ablation::AblationComparison {
                cryo_contrast_significant: true,
                uv_response_significant: true,
                cryo_effect_size: 4.0,
                uv_effect_size: 0.4,
                interpretation: "Test interpretation".to_string(),
            },
        };

        let html = HtmlReport::render_html(&config, &sites, &ablation, None).unwrap();

        assert!(html.contains("PRISM4D"));
        assert!(html.contains("Ablation"));
        assert!(html.contains("mandatory"));
    }
}
