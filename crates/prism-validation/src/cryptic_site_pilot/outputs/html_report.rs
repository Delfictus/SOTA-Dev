//! HTML Report Generator for Cryptic Site Pilot
//!
//! Generates executive-quality HTML reports for pharmaceutical partners.
//! Includes inline CSS (no external dependencies).

use crate::cryptic_site_pilot::druggability::{DruggabilityClass, DruggabilityScore};
use crate::cryptic_site_pilot::volume_tracker::{VolumeStatistics, VolumeTimeSeries};
use chrono::{DateTime, Utc};
use std::io::Write;

/// Cryptic site data for report
pub struct CrypticSiteReport {
    pub site_id: String,
    pub rank: usize,
    pub residues: Vec<i32>,
    pub centroid: [f64; 3],
    pub volume_stats: VolumeStatistics,
    pub druggability: DruggabilityScore,
    pub representative_frame: usize,
}

/// Simulation metadata for report
pub struct SimulationMetadata {
    pub pdb_id: String,
    pub n_residues: usize,
    pub n_atoms: usize,
    pub n_frames: usize,
    pub temperature_k: f32,
    pub duration_ns: f64,
    pub mean_rmsd: f64,
    pub rmsd_std: f64,
}

/// HTML Report Generator
pub struct ReportGenerator {
    /// Report title
    pub title: String,
    /// Company/Lab name
    pub organization: String,
    /// Analysis timestamp
    pub timestamp: DateTime<Utc>,
}

impl Default for ReportGenerator {
    fn default() -> Self {
        Self {
            title: "PRISM-4D Cryptic Binding Site Analysis".to_string(),
            organization: "PRISM-4D".to_string(),
            timestamp: Utc::now(),
        }
    }
}

impl ReportGenerator {
    pub fn new(title: &str) -> Self {
        Self {
            title: title.to_string(),
            ..Default::default()
        }
    }

    /// Generate complete HTML report
    pub fn generate<W: Write>(
        &self,
        writer: &mut W,
        metadata: &SimulationMetadata,
        cryptic_sites: &[CrypticSiteReport],
    ) -> std::io::Result<()> {
        // Header
        self.write_header(writer)?;

        // Title section
        self.write_title_section(writer, metadata)?;

        // Simulation summary
        self.write_simulation_summary(writer, metadata)?;

        // Cryptic sites detected
        self.write_cryptic_sites_section(writer, cryptic_sites)?;

        // Detailed per-site analysis
        for site in cryptic_sites {
            self.write_site_detail(writer, site)?;
        }

        // Deliverables section
        self.write_deliverables_section(writer, metadata)?;

        // Footer
        self.write_footer(writer)?;

        Ok(())
    }

    fn write_header<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        write!(writer, r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{}</title>
    <style>
        :root {{
            --primary: #2563eb;
            --success: #22c55e;
            --warning: #eab308;
            --danger: #ef4444;
            --dark: #1e293b;
            --light: #f8fafc;
            --border: #e2e8f0;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            color: var(--dark);
            background: var(--light);
        }}

        .container {{
            max-width: 1000px;
            margin: 0 auto;
            padding: 40px 20px;
        }}

        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 3px solid var(--primary);
        }}

        .header h1 {{
            font-size: 28px;
            color: var(--dark);
            margin-bottom: 10px;
        }}

        .header .subtitle {{
            font-size: 18px;
            color: #64748b;
        }}

        .section {{
            background: white;
            border-radius: 8px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border: 1px solid var(--border);
        }}

        .section h2 {{
            font-size: 20px;
            color: var(--dark);
            margin-bottom: 16px;
            padding-bottom: 8px;
            border-bottom: 2px solid var(--primary);
        }}

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

        .stat-card .label {{
            font-size: 12px;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .stat-card .value {{
            font-size: 28px;
            font-weight: 700;
            color: var(--dark);
            margin: 4px 0;
        }}

        .stat-card .unit {{
            font-size: 14px;
            color: #94a3b8;
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
            border-bottom: 1px solid var(--border);
        }}

        .site-name {{
            font-weight: 600;
            font-size: 16px;
        }}

        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
        }}

        .badge-success {{ background: #dcfce7; color: #166534; }}
        .badge-warning {{ background: #fef9c3; color: #854d0e; }}
        .badge-danger {{ background: #fee2e2; color: #991b1b; }}
        .badge-info {{ background: #dbeafe; color: #1e40af; }}

        .site-body {{
            padding: 16px;
        }}

        .site-stats {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            margin-bottom: 16px;
        }}

        .mini-stat {{
            text-align: center;
        }}

        .mini-stat .label {{
            font-size: 11px;
            color: #64748b;
        }}

        .mini-stat .value {{
            font-size: 18px;
            font-weight: 600;
        }}

        .residue-list {{
            font-family: monospace;
            font-size: 13px;
            background: var(--light);
            padding: 8px 12px;
            border-radius: 4px;
            word-break: break-all;
        }}

        .meter {{
            height: 8px;
            background: #e2e8f0;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 8px;
        }}

        .meter-fill {{
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s ease;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
        }}

        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}

        th {{
            background: var(--light);
            font-weight: 600;
            font-size: 12px;
            text-transform: uppercase;
            color: #64748b;
        }}

        .footer {{
            text-align: center;
            padding-top: 20px;
            border-top: 1px solid var(--border);
            color: #64748b;
            font-size: 14px;
        }}

        .highlight {{
            background: linear-gradient(120deg, #dbeafe 0%, #dbeafe 100%);
            padding: 2px 4px;
            border-radius: 3px;
        }}

        @media print {{
            body {{ background: white; }}
            .section {{ box-shadow: none; }}
        }}
    </style>
</head>
<body>
    <div class="container">
"#, self.title)
    }

    fn write_title_section<W: Write>(&self, writer: &mut W, metadata: &SimulationMetadata) -> std::io::Result<()> {
        write!(writer, r#"
        <div class="header">
            <h1>{}</h1>
            <div class="subtitle">Target: <strong>{}</strong> | Generated: {}</div>
        </div>
"#, self.title, metadata.pdb_id, self.timestamp.format("%Y-%m-%d %H:%M UTC"))
    }

    fn write_simulation_summary<W: Write>(&self, writer: &mut W, metadata: &SimulationMetadata) -> std::io::Result<()> {
        write!(writer, r#"
        <div class="section">
            <h2>Simulation Summary</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="label">Duration</div>
                    <div class="value">{:.1}</div>
                    <div class="unit">nanoseconds</div>
                </div>
                <div class="stat-card">
                    <div class="label">Temperature</div>
                    <div class="value">{:.1}</div>
                    <div class="unit">Kelvin</div>
                </div>
                <div class="stat-card">
                    <div class="label">Frames Analyzed</div>
                    <div class="value">{}</div>
                    <div class="unit">conformations</div>
                </div>
                <div class="stat-card">
                    <div class="label">RMSD Stability</div>
                    <div class="value">{:.2} &plusmn; {:.2}</div>
                    <div class="unit">Angstroms</div>
                </div>
            </div>
        </div>
"#, metadata.duration_ns, metadata.temperature_k, metadata.n_frames, metadata.mean_rmsd, metadata.rmsd_std)
    }

    fn write_cryptic_sites_section<W: Write>(&self, writer: &mut W, sites: &[CrypticSiteReport]) -> std::io::Result<()> {
        write!(writer, r#"
        <div class="section">
            <h2>Cryptic Sites Detected: {}</h2>
"#, sites.len())?;

        if sites.is_empty() {
            write!(writer, r#"
            <p style="color: #64748b; font-style: italic;">No cryptic binding sites detected in this structure.</p>
"#)?;
        } else {
            write!(writer, r#"
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Site ID</th>
                        <th>Residues</th>
                        <th>Avg Volume</th>
                        <th>Open Freq</th>
                        <th>Druggability</th>
                    </tr>
                </thead>
                <tbody>
"#)?;

            for site in sites {
                let drug_class = site.druggability.classification;
                let badge_class = match drug_class {
                    DruggabilityClass::HighlyDruggable => "badge-success",
                    DruggabilityClass::Druggable => "badge-info",
                    DruggabilityClass::Challenging => "badge-warning",
                    DruggabilityClass::Difficult => "badge-danger",
                };

                write!(writer, r#"
                    <tr>
                        <td><strong>#{}</strong></td>
                        <td>{}</td>
                        <td>{}</td>
                        <td>{:.0} ų</td>
                        <td>{:.0}%</td>
                        <td><span class="badge {}">{}</span></td>
                    </tr>
"#,
                    site.rank,
                    site.site_id,
                    site.residues.len(),
                    site.volume_stats.mean_volume,
                    site.volume_stats.open_frequency * 100.0,
                    badge_class,
                    drug_class.name()
                )?;
            }

            write!(writer, r#"
                </tbody>
            </table>
"#)?;
        }

        write!(writer, r#"
        </div>
"#)
    }

    fn write_site_detail<W: Write>(&self, writer: &mut W, site: &CrypticSiteReport) -> std::io::Result<()> {
        let drug = &site.druggability;
        let stats = &site.volume_stats;

        let drug_color = drug.classification.color();

        write!(writer, r#"
        <div class="section">
            <h2>Site {}: "{}"</h2>
            <div class="site-card">
                <div class="site-header">
                    <span class="site-name">Cryptic Pocket Analysis</span>
                    <span class="badge" style="background: {}20; color: {};">Druggability: {:.2}</span>
                </div>
                <div class="site-body">
                    <div class="site-stats">
                        <div class="mini-stat">
                            <div class="label">Volume (open)</div>
                            <div class="value">{:.0} ų</div>
                        </div>
                        <div class="mini-stat">
                            <div class="label">Open Frequency</div>
                            <div class="value">{:.0}%</div>
                        </div>
                        <div class="mini-stat">
                            <div class="label">CV (variance)</div>
                            <div class="value">{:.0}%</div>
                        </div>
                        <div class="mini-stat">
                            <div class="label">Representative</div>
                            <div class="value">Frame {}</div>
                        </div>
                    </div>

                    <p><strong>Location:</strong></p>
                    <div class="residue-list">{}</div>

                    <p style="margin-top: 16px;"><strong>Druggability Components:</strong></p>
                    <div style="margin-top: 8px;">
                        <div style="display: flex; justify-content: space-between; font-size: 13px;">
                            <span>Hydrophobicity</span>
                            <span>{:.2}</span>
                        </div>
                        <div class="meter">
                            <div class="meter-fill" style="width: {:.0}%; background: var(--primary);"></div>
                        </div>
                    </div>
                    <div style="margin-top: 8px;">
                        <div style="display: flex; justify-content: space-between; font-size: 13px;">
                            <span>Enclosure</span>
                            <span>{:.2}</span>
                        </div>
                        <div class="meter">
                            <div class="meter-fill" style="width: {:.0}%; background: var(--success);"></div>
                        </div>
                    </div>
                    <div style="margin-top: 8px;">
                        <div style="display: flex; justify-content: space-between; font-size: 13px;">
                            <span>H-bond Capacity</span>
                            <span>{:.2}</span>
                        </div>
                        <div class="meter">
                            <div class="meter-fill" style="width: {:.0}%; background: var(--warning);"></div>
                        </div>
                    </div>

                    <p style="margin-top: 16px;"><strong>Estimated Binding Affinity:</strong>
                        <span class="highlight">{:.1} to {:.1} kcal/mol</span>
                    </p>
                    <p><strong>Recommended Fragment Size:</strong>
                        <span class="highlight">{}-{} heavy atoms</span>
                    </p>
                </div>
            </div>
        </div>
"#,
            site.rank,
            site.site_id,
            drug_color,
            drug_color,
            drug.score,
            stats.mean_volume,
            stats.open_frequency * 100.0,
            stats.cv_volume * 100.0,
            site.representative_frame,
            site.residues.iter().map(|r| r.to_string()).collect::<Vec<_>>().join(", "),
            drug.hydrophobicity,
            drug.hydrophobicity * 100.0,
            drug.enclosure,
            drug.enclosure * 100.0,
            drug.hbond_capacity,
            drug.hbond_capacity * 100.0,
            drug.estimated_affinity_range.0,
            drug.estimated_affinity_range.1,
            drug.recommended_fragment_size.0,
            drug.recommended_fragment_size.1
        )
    }

    fn write_deliverables_section<W: Write>(&self, writer: &mut W, metadata: &SimulationMetadata) -> std::io::Result<()> {
        write!(writer, r#"
        <div class="section">
            <h2>Deliverables</h2>
            <table>
                <thead>
                    <tr>
                        <th>File</th>
                        <th>Format</th>
                        <th>Description</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><code>{}_trajectory.pdb</code></td>
                        <td>Multi-MODEL PDB</td>
                        <td>Full trajectory ({} frames)</td>
                    </tr>
                    <tr>
                        <td><code>{}_sites.pdb</code></td>
                        <td>Multi-MODEL PDB</td>
                        <td>Top 5 open conformations per site</td>
                    </tr>
                    <tr>
                        <td><code>{}_rmsf.csv</code></td>
                        <td>CSV</td>
                        <td>Per-residue RMSF values</td>
                    </tr>
                    <tr>
                        <td><code>{}_volumes.csv</code></td>
                        <td>CSV</td>
                        <td>Pocket volume time series</td>
                    </tr>
                    <tr>
                        <td><code>{}_contacts.csv</code></td>
                        <td>CSV</td>
                        <td>Binding site residue contacts</td>
                    </tr>
                </tbody>
            </table>
        </div>
"#,
            metadata.pdb_id,
            metadata.n_frames,
            metadata.pdb_id,
            metadata.pdb_id,
            metadata.pdb_id,
            metadata.pdb_id
        )
    }

    fn write_footer<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        write!(writer, r#"
        <div class="footer">
            <p>Generated by <strong>PRISM-4D</strong> Cryptic Binding Site Detection Pipeline</p>
            <p>&copy; {} {}</p>
        </div>
    </div>
</body>
</html>
"#, self.timestamp.format("%Y"), self.organization)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cryptic_site_pilot::druggability::DruggabilityScorer;

    #[test]
    fn test_generate_report() {
        let generator = ReportGenerator::new("Test Analysis");

        let metadata = SimulationMetadata {
            pdb_id: "1BTL".to_string(),
            n_residues: 263,
            n_atoms: 2100,
            n_frames: 200,
            temperature_k: 310.0,
            duration_ns: 10.0,
            mean_rmsd: 1.2,
            rmsd_std: 0.3,
        };

        let scorer = DruggabilityScorer::default();
        let residue_names: Vec<String> = vec!["LEU", "ILE", "VAL", "PHE", "SER"]
            .into_iter().map(String::from).collect();
        let drug_score = scorer.score_simple(&residue_names, 300.0);

        let sites = vec![CrypticSiteReport {
            site_id: "Omega loop".to_string(),
            rank: 1,
            residues: vec![165, 166, 167, 168, 169, 170],
            centroid: [10.0, 20.0, 30.0],
            volume_stats: VolumeStatistics {
                n_frames: 200,
                n_open_frames: 46,
                open_frequency: 0.23,
                mean_volume: 287.0,
                std_volume: 85.0,
                cv_volume: 0.30,
                min_volume: 120.0,
                max_volume: 450.0,
                breathing_amplitude: 330.0,
                mean_sasa: 200.0,
                mean_druggability: Some(0.78),
                max_volume_frame: 4521,
                min_volume_frame: 100,
            },
            druggability: drug_score,
            representative_frame: 4521,
        }];

        let mut output = Vec::new();
        generator.generate(&mut output, &metadata, &sites).unwrap();

        let html = String::from_utf8(output).unwrap();
        assert!(html.contains("1BTL"));
        assert!(html.contains("Omega loop"));
        assert!(html.contains("287"));
        assert!(html.contains("23%"));
    }

    #[test]
    fn test_empty_sites() {
        let generator = ReportGenerator::default();

        let metadata = SimulationMetadata {
            pdb_id: "TEST".to_string(),
            n_residues: 100,
            n_atoms: 800,
            n_frames: 50,
            temperature_k: 300.0,
            duration_ns: 5.0,
            mean_rmsd: 1.0,
            rmsd_std: 0.2,
        };

        let mut output = Vec::new();
        generator.generate(&mut output, &metadata, &[]).unwrap();

        let html = String::from_utf8(output).unwrap();
        assert!(html.contains("Cryptic Sites Detected: 0"));
        assert!(html.contains("No cryptic binding sites detected"));
    }
}
