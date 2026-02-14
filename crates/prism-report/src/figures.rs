//! Figure generation using plotters (SVG output)
//!
//! Uses SVG backend to avoid system font dependencies.

use crate::ablation::AblationResults;
use crate::sites::CrypticSite;
use anyhow::Result;
use plotters::prelude::*;
use plotters_svg::SVGBackend;
use std::path::Path;

/// Generate pocket volume vs time figure
pub fn generate_volume_vs_time(
    path: &Path,
    volumes: &[(f32, f64)], // (time_ps, volume)
    site_id: &str,
) -> Result<()> {
    // Convert path to SVG if it's PNG
    let svg_path = if path.extension().map(|e| e == "png").unwrap_or(false) {
        path.with_extension("svg")
    } else {
        path.to_path_buf()
    };

    let root = SVGBackend::new(&svg_path, (800, 500)).into_drawing_area();
    root.fill(&WHITE)?;

    if volumes.is_empty() {
        root.draw(&Text::new(
            "No volume data",
            (400, 250),
            ("sans-serif", 20).into_font().color(&BLACK),
        ))?;
        root.present()?;
        return Ok(());
    }

    let (min_time, max_time) = volumes
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), (t, _)| {
            (min.min(*t), max.max(*t))
        });

    let (min_vol, max_vol) = volumes
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), (_, v)| {
            (min.min(*v), max.max(*v))
        });

    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("{} - Pocket Volume vs Time", site_id),
            ("sans-serif", 20),
        )
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(min_time..max_time, min_vol..max_vol * 1.1)?;

    chart
        .configure_mesh()
        .x_desc("Time (ps)")
        .y_desc("Volume (A^3)")
        .draw()?;

    chart.draw_series(LineSeries::new(
        volumes.iter().map(|(t, v)| (*t, *v)),
        &BLUE,
    ))?;

    root.present()?;
    Ok(())
}

/// Generate persistence vs replica heatmap
pub fn generate_persistence_vs_replica(
    path: &Path,
    data: &[(String, Vec<f64>)], // (site_id, [replica_persistence...])
) -> Result<()> {
    let svg_path = if path.extension().map(|e| e == "png").unwrap_or(false) {
        path.with_extension("svg")
    } else {
        path.to_path_buf()
    };

    let root = SVGBackend::new(&svg_path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    if data.is_empty() {
        root.draw(&Text::new(
            "No data available",
            (400, 300),
            ("sans-serif", 20).into_font().color(&BLACK),
        ))?;
        root.present()?;
        return Ok(());
    }

    let n_sites = data.len();
    let n_replicates = data[0].1.len().max(1);

    let mut chart = ChartBuilder::on(&root)
        .caption("Persistence vs Replica Agreement", ("sans-serif", 20))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(100)
        .build_cartesian_2d(0..n_replicates, 0..n_sites)?;

    chart
        .configure_mesh()
        .x_desc("Replica")
        .y_desc("Site")
        .y_label_formatter(&|y| {
            if *y < data.len() {
                data[*y].0.clone()
            } else {
                String::new()
            }
        })
        .draw()?;

    // Draw heatmap cells
    for (site_idx, (_, persistences)) in data.iter().enumerate() {
        for (rep_idx, &persistence) in persistences.iter().enumerate() {
            let color = heatmap_color(persistence);
            chart.draw_series(std::iter::once(Rectangle::new(
                [(rep_idx, site_idx), (rep_idx + 1, site_idx + 1)],
                color.filled(),
            )))?;
        }
    }

    root.present()?;
    Ok(())
}

/// Generate UV vs control delta SASA bar chart
pub fn generate_uv_vs_control_deltasasa(
    path: &Path,
    sites: &[CrypticSite],
) -> Result<()> {
    let svg_path = if path.extension().map(|e| e == "png").unwrap_or(false) {
        path.with_extension("svg")
    } else {
        path.to_path_buf()
    };

    let root = SVGBackend::new(&svg_path, (800, 500)).into_drawing_area();
    root.fill(&WHITE)?;

    if sites.is_empty() {
        root.draw(&Text::new(
            "No sites to display",
            (400, 250),
            ("sans-serif", 20).into_font().color(&BLACK),
        ))?;
        root.present()?;
        return Ok(());
    }

    let data: Vec<_> = sites
        .iter()
        .map(|s| (s.site_id.clone(), s.metrics.uv_response.delta_sasa))
        .collect();

    let max_abs = data
        .iter()
        .map(|(_, v)| v.abs())
        .fold(0.0f64, f64::max)
        .max(1.0);

    let mut chart = ChartBuilder::on(&root)
        .caption("UV Response: Delta SASA (cryo+UV - cryo-only)", ("sans-serif", 18))
        .margin(20)
        .x_label_area_size(80)
        .y_label_area_size(60)
        .build_cartesian_2d(
            (0..data.len()).into_segmented(),
            -max_abs * 1.1..max_abs * 1.1,
        )?;

    chart
        .configure_mesh()
        .x_labels(data.len())
        .x_label_formatter(&|x| {
            if let plotters::prelude::SegmentValue::CenterOf(idx) = x {
                if *idx < data.len() {
                    return data[*idx].0.clone();
                }
            }
            String::new()
        })
        .y_desc("Delta SASA (A^2)")
        .draw()?;

    chart.draw_series(data.iter().enumerate().map(|(idx, (_, val))| {
        let color = if *val >= 0.0 { &GREEN } else { &RED };
        let y0 = 0.0f64;
        let y1 = *val;
        Rectangle::new(
            [
                (SegmentValue::CenterOf(idx), y0),
                (SegmentValue::CenterOf(idx + 1), y1),
            ],
            color.filled(),
        )
    }))?;

    // Zero line
    chart.draw_series(LineSeries::new(
        [(SegmentValue::Exact(0), 0.0), (SegmentValue::Exact(data.len()), 0.0)],
        &BLACK,
    ))?;

    root.present()?;
    Ok(())
}

/// Generate holo distance histogram (if holo provided)
pub fn generate_holo_distance_hist(
    path: &Path,
    distances: &[f32],
    site_id: &str,
) -> Result<()> {
    let svg_path = if path.extension().map(|e| e == "png").unwrap_or(false) {
        path.with_extension("svg")
    } else {
        path.to_path_buf()
    };

    let root = SVGBackend::new(&svg_path, (800, 500)).into_drawing_area();
    root.fill(&WHITE)?;

    if distances.is_empty() {
        root.draw(&Text::new(
            "No distance data",
            (400, 250),
            ("sans-serif", 20).into_font().color(&BLACK),
        ))?;
        root.present()?;
        return Ok(());
    }

    // Create histogram bins manually
    let max_dist = distances.iter().fold(0.0f32, |a, &b| a.max(b)).max(15.0);
    let n_bins = 20usize;
    let bin_width = max_dist / n_bins as f32;

    let mut bins = vec![0u32; n_bins];
    for &d in distances {
        let bin = ((d / bin_width) as usize).min(n_bins - 1);
        bins[bin] += 1;
    }

    let max_count = *bins.iter().max().unwrap_or(&1);

    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("{} - Distance to Ligand", site_id),
            ("sans-serif", 20),
        )
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0f32..max_dist, 0u32..max_count + 1)?;

    chart
        .configure_mesh()
        .x_desc("Distance (A)")
        .y_desc("Count")
        .draw()?;

    // Draw histogram bars manually
    for (i, &count) in bins.iter().enumerate() {
        let x0 = i as f32 * bin_width;
        let x1 = (i + 1) as f32 * bin_width;
        chart.draw_series(std::iter::once(Rectangle::new(
            [(x0, 0), (x1, count)],
            BLUE.filled(),
        )))?;
    }

    // Mark 5A threshold
    chart.draw_series(LineSeries::new(
        [(5.0, 0), (5.0, max_count)],
        RED.stroke_width(2),
    ))?;

    root.present()?;
    Ok(())
}

/// Generate pocket overlay figure (simple schematic)
pub fn generate_pocket_overlay(
    path: &Path,
    site: &CrypticSite,
    _ablation: Option<&AblationResults>,
) -> Result<()> {
    let svg_path = if path.extension().map(|e| e == "png").unwrap_or(false) {
        path.with_extension("svg")
    } else {
        path.to_path_buf()
    };

    let root = SVGBackend::new(&svg_path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    // Draw a schematic representation
    let cx = 400.0f32;
    let cy = 300.0f32;

    // Title
    root.draw(&Text::new(
        format!("{} - Pocket Overview", site.site_id),
        (20, 20),
        ("sans-serif", 20).into_font().color(&BLACK),
    ))?;

    // Draw protein outline (placeholder circle)
    root.draw(&Circle::new(
        (cx as i32, cy as i32),
        150,
        ShapeStyle::from(&RGBColor(200, 200, 200)).filled(),
    ))?;

    // Draw pocket region
    let pocket_radius = (site.metrics.geometry.volume_mean.sqrt() * 2.0) as i32;
    root.draw(&Circle::new(
        (cx as i32, cy as i32),
        pocket_radius.min(100),
        ShapeStyle::from(&RGBColor(255, 105, 180)).filled(), // Hot pink
    ))?;

    // Centroid marker
    root.draw(&Circle::new(
        (cx as i32, cy as i32),
        5,
        ShapeStyle::from(&YELLOW).filled(),
    ))?;

    // Info text
    let info = format!(
        "Volume: {:.0} A^3\nResidues: {}\nHydrophobic: {:.0}%\nConfidence: {:.2}",
        site.metrics.geometry.volume_mean,
        site.residues.len(),
        site.metrics.chemistry.hydrophobic_fraction * 100.0,
        site.confidence
    );

    let lines: Vec<&str> = info.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        root.draw(&Text::new(
            *line,
            (550, 200 + i as i32 * 25),
            ("sans-serif", 16).into_font().color(&BLACK),
        ))?;
    }

    root.present()?;
    Ok(())
}

/// Helper: map value [0, 1] to heatmap color
fn heatmap_color(value: f64) -> RGBColor {
    let v = value.clamp(0.0, 1.0);
    // Blue (low) -> White (mid) -> Red (high)
    if v < 0.5 {
        let t = v * 2.0;
        RGBColor(
            (255.0 * t) as u8,
            (255.0 * t) as u8,
            255,
        )
    } else {
        let t = (v - 0.5) * 2.0;
        RGBColor(
            255,
            (255.0 * (1.0 - t)) as u8,
            (255.0 * (1.0 - t)) as u8,
        )
    }
}

// =============================================================================
// REQUIRED CRYPTIC SITE FIGURES (ALL SIX)
// =============================================================================

/// Site data for figures (with real metrics)
pub struct SiteFigureData {
    pub site_id: String,
    pub open_frequency: f64,
    pub cv_sasa: f64,
    pub volume_mean: f64,
    pub residue_count: usize,
    pub dynamics_level: String, // "Low", "Moderate", "High"
}

impl SiteFigureData {
    pub fn from_site(site: &CrypticSite) -> Self {
        // Compute CV_SASA proxy from volume variance
        let cv_sasa = if site.metrics.geometry.volume_mean > 0.0 {
            (site.metrics.geometry.volume_p95 - site.metrics.geometry.volume_p50).abs()
                / site.metrics.geometry.volume_mean
        } else {
            0.0
        };

        let dynamics_level = if cv_sasa < 0.3 {
            "Low"
        } else if cv_sasa < 0.6 {
            "Moderate"
        } else {
            "High"
        };

        Self {
            site_id: site.site_id.clone(),
            open_frequency: site.metrics.persistence.present_fraction,
            cv_sasa,
            volume_mean: site.metrics.geometry.volume_mean,
            residue_count: site.residues.len(),
            dynamics_level: dynamics_level.to_string(),
        }
    }
}

/// Generate all six required cryptic site figures
pub fn generate_cryptic_site_figures(
    figures_dir: &Path,
    sites: &[CrypticSite],
) -> Result<()> {
    let site_data: Vec<SiteFigureData> = sites.iter().map(SiteFigureData::from_site).collect();

    // TOP ROW
    // 1. Distribution by Dynamics Level (bar chart)
    generate_dynamics_distribution(
        &figures_dir.join("01_dynamics_distribution.svg"),
        &site_data,
    )?;

    // 2. Cryptic Site Classification (scatter: OpenFreq vs CV_SASA)
    generate_cryptic_classification(
        &figures_dir.join("02_cryptic_classification.svg"),
        &site_data,
    )?;

    // MIDDLE ROW
    // 3. CV SASA Distribution (histogram)
    generate_cv_sasa_distribution(
        &figures_dir.join("03_cv_sasa_histogram.svg"),
        &site_data,
    )?;

    // 4. Volume vs Dynamics (scatter: OpenFreq vs Volume)
    generate_volume_vs_dynamics(
        &figures_dir.join("04_volume_vs_dynamics.svg"),
        &site_data,
    )?;

    // BOTTOM ROW
    // 5. Opening Frequency Distribution (histogram)
    generate_open_frequency_distribution(
        &figures_dir.join("05_open_frequency_histogram.svg"),
        &site_data,
    )?;

    // 6. Pocket Size vs Dynamics (scatter: ResidueCount vs CV_SASA)
    generate_pocket_size_vs_dynamics(
        &figures_dir.join("06_pocket_size_vs_dynamics.svg"),
        &site_data,
    )?;

    Ok(())
}

/// Figure 1: Distribution by Dynamics Level (bar chart)
/// X: Low / Moderate / High, Y: Number of pockets
pub fn generate_dynamics_distribution(
    path: &Path,
    sites: &[SiteFigureData],
) -> Result<()> {
    let root = SVGBackend::new(path, (600, 400)).into_drawing_area();
    root.fill(&WHITE)?;

    // Count by dynamics level
    let mut low = 0usize;
    let mut moderate = 0usize;
    let mut high = 0usize;

    for site in sites {
        match site.dynamics_level.as_str() {
            "Low" => low += 1,
            "Moderate" => moderate += 1,
            "High" => high += 1,
            _ => {}
        }
    }

    let max_count = low.max(moderate).max(high).max(1) as u32;
    let data = vec![("Low", low as u32), ("Moderate", moderate as u32), ("High", high as u32)];

    let mut chart = ChartBuilder::on(&root)
        .caption("Distribution by Dynamics Level", ("sans-serif", 18))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(
            (0..3).into_segmented(),
            0u32..max_count + 2,
        )?;

    chart
        .configure_mesh()
        .x_labels(3)
        .x_label_formatter(&|x| {
            match x {
                SegmentValue::CenterOf(0) => "Low".to_string(),
                SegmentValue::CenterOf(1) => "Moderate".to_string(),
                SegmentValue::CenterOf(2) => "High".to_string(),
                _ => String::new(),
            }
        })
        .y_desc("Number of Pockets")
        .draw()?;

    // Draw bars with colors
    let colors = [&BLUE, &YELLOW, &RED];
    for (i, (_, count)) in data.iter().enumerate() {
        chart.draw_series(std::iter::once(Rectangle::new(
            [(SegmentValue::CenterOf(i as i32), 0), (SegmentValue::CenterOf((i + 1) as i32), *count)],
            colors[i].filled(),
        )))?;
    }

    root.present()?;
    Ok(())
}

/// Figure 2: Cryptic Site Classification (scatter plot)
/// X: Open Frequency, Y: CV SASA, color by dynamics, threshold line at CV=0.2
pub fn generate_cryptic_classification(
    path: &Path,
    sites: &[SiteFigureData],
) -> Result<()> {
    let root = SVGBackend::new(path, (700, 500)).into_drawing_area();
    root.fill(&WHITE)?;

    if sites.is_empty() {
        root.draw(&Text::new(
            "No sites to display",
            (350, 250),
            ("sans-serif", 20).into_font().color(&BLACK),
        ))?;
        root.present()?;
        return Ok(());
    }

    let max_open = sites.iter().map(|s| s.open_frequency).fold(0.0f64, f64::max).max(1.0);
    let max_cv = sites.iter().map(|s| s.cv_sasa).fold(0.0f64, f64::max).max(1.0);

    let mut chart = ChartBuilder::on(&root)
        .caption("Cryptic Site Classification", ("sans-serif", 18))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0.0..max_open * 1.1, 0.0..max_cv * 1.1)?;

    chart
        .configure_mesh()
        .x_desc("Open Frequency")
        .y_desc("CV SASA")
        .draw()?;

    // Cryptic threshold line at CV = 0.2
    chart.draw_series(LineSeries::new(
        [(0.0, 0.2), (max_open * 1.1, 0.2)],
        RED.stroke_width(2),
    ))?;

    // Label the threshold
    root.draw(&Text::new(
        "Cryptic threshold (CV=0.2)",
        (450, 80),
        ("sans-serif", 12).into_font().color(&RED),
    ))?;

    // Plot points colored by dynamics level
    for site in sites {
        let color = match site.dynamics_level.as_str() {
            "Low" => BLUE,
            "Moderate" => YELLOW,
            "High" => RED,
            _ => BLACK,
        };
        chart.draw_series(std::iter::once(Circle::new(
            (site.open_frequency, site.cv_sasa),
            5,
            ShapeStyle::from(&color).filled(),
        )))?;
    }

    // Label highest-CV pocket
    if let Some(highest) = sites.iter().max_by(|a, b| a.cv_sasa.partial_cmp(&b.cv_sasa).unwrap()) {
        root.draw(&Text::new(
            highest.site_id.clone(),
            ((highest.open_frequency * 500.0 / max_open + 100.0) as i32,
             (400.0 - highest.cv_sasa * 350.0 / max_cv) as i32),
            ("sans-serif", 10).into_font().color(&BLACK),
        ))?;
    }

    root.present()?;
    Ok(())
}

/// Figure 3: CV SASA Distribution (histogram)
pub fn generate_cv_sasa_distribution(
    path: &Path,
    sites: &[SiteFigureData],
) -> Result<()> {
    let root = SVGBackend::new(path, (600, 400)).into_drawing_area();
    root.fill(&WHITE)?;

    if sites.is_empty() {
        root.draw(&Text::new("No data", (300, 200), ("sans-serif", 20).into_font().color(&BLACK)))?;
        root.present()?;
        return Ok(());
    }

    let cv_values: Vec<f64> = sites.iter().map(|s| s.cv_sasa).collect();
    let max_cv = cv_values.iter().fold(0.0f64, |a, &b| a.max(b)).max(1.0);
    let mean_cv = cv_values.iter().sum::<f64>() / cv_values.len() as f64;

    // Create histogram bins
    let n_bins = 15usize;
    let bin_width = max_cv / n_bins as f64;
    let mut bins = vec![0u32; n_bins];
    for &cv in &cv_values {
        let bin = ((cv / bin_width) as usize).min(n_bins - 1);
        bins[bin] += 1;
    }
    let max_count = *bins.iter().max().unwrap_or(&1);

    let mut chart = ChartBuilder::on(&root)
        .caption("CV SASA Distribution", ("sans-serif", 18))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0.0f64..max_cv, 0u32..max_count + 2)?;

    chart
        .configure_mesh()
        .x_desc("CV SASA")
        .y_desc("Count")
        .draw()?;

    // Draw histogram bars
    for (i, &count) in bins.iter().enumerate() {
        let x0 = i as f64 * bin_width;
        let x1 = (i + 1) as f64 * bin_width;
        chart.draw_series(std::iter::once(Rectangle::new(
            [(x0, 0), (x1, count)],
            BLUE.filled(),
        )))?;
    }

    // Mark mean
    chart.draw_series(LineSeries::new(
        [(mean_cv, 0), (mean_cv, max_count)],
        GREEN.stroke_width(2),
    ))?;

    // Mark cryptic threshold at 0.2
    if 0.2 < max_cv {
        chart.draw_series(LineSeries::new(
            [(0.2, 0), (0.2, max_count)],
            RED.stroke_width(2),
        ))?;
    }

    root.present()?;
    Ok(())
}

/// Figure 4: Volume vs Dynamics (scatter plot)
/// X: Open Frequency, Y: Mean Volume, color gradient by CV SASA
pub fn generate_volume_vs_dynamics(
    path: &Path,
    sites: &[SiteFigureData],
) -> Result<()> {
    let root = SVGBackend::new(path, (700, 500)).into_drawing_area();
    root.fill(&WHITE)?;

    if sites.is_empty() {
        root.draw(&Text::new("No data", (350, 250), ("sans-serif", 20).into_font().color(&BLACK)))?;
        root.present()?;
        return Ok(());
    }

    let max_open = sites.iter().map(|s| s.open_frequency).fold(0.0f64, f64::max).max(1.0);
    let max_vol = sites.iter().map(|s| s.volume_mean).fold(0.0f64, f64::max).max(100.0);
    let max_cv = sites.iter().map(|s| s.cv_sasa).fold(0.0f64, f64::max).max(1.0);

    let mut chart = ChartBuilder::on(&root)
        .caption("Volume vs Dynamics", ("sans-serif", 18))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0..max_open * 1.1, 0.0..max_vol * 1.1)?;

    chart
        .configure_mesh()
        .x_desc("Open Frequency")
        .y_desc("Mean Volume (A^3)")
        .draw()?;

    // Plot points with color gradient by CV SASA
    for site in sites {
        let t = if max_cv > 0.0 { site.cv_sasa / max_cv } else { 0.0 };
        let color = RGBColor(
            (255.0 * t) as u8,
            0,
            (255.0 * (1.0 - t)) as u8,
        );
        chart.draw_series(std::iter::once(Circle::new(
            (site.open_frequency, site.volume_mean),
            6,
            ShapeStyle::from(&color).filled(),
        )))?;
    }

    root.present()?;
    Ok(())
}

/// Figure 5: Opening Frequency Distribution (histogram)
pub fn generate_open_frequency_distribution(
    path: &Path,
    sites: &[SiteFigureData],
) -> Result<()> {
    let root = SVGBackend::new(path, (600, 400)).into_drawing_area();
    root.fill(&WHITE)?;

    if sites.is_empty() {
        root.draw(&Text::new("No data", (300, 200), ("sans-serif", 20).into_font().color(&BLACK)))?;
        root.present()?;
        return Ok(());
    }

    let open_values: Vec<f64> = sites.iter().map(|s| s.open_frequency).collect();
    let max_open = 1.0f64; // Always 0-1 range
    let mean_open = open_values.iter().sum::<f64>() / open_values.len() as f64;

    // Create histogram bins
    let n_bins = 10usize;
    let bin_width = max_open / n_bins as f64;
    let mut bins = vec![0u32; n_bins];
    for &open in &open_values {
        let bin = ((open / bin_width) as usize).min(n_bins - 1);
        bins[bin] += 1;
    }
    let max_count = *bins.iter().max().unwrap_or(&1);

    let mut chart = ChartBuilder::on(&root)
        .caption("Opening Frequency Distribution", ("sans-serif", 18))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0.0f64..1.0, 0u32..max_count + 2)?;

    chart
        .configure_mesh()
        .x_desc("Opening Frequency")
        .y_desc("Count")
        .draw()?;

    // Draw histogram bars
    for (i, &count) in bins.iter().enumerate() {
        let x0 = i as f64 * bin_width;
        let x1 = (i + 1) as f64 * bin_width;
        chart.draw_series(std::iter::once(Rectangle::new(
            [(x0, 0), (x1, count)],
            CYAN.filled(),
        )))?;
    }

    // Mark mean
    chart.draw_series(LineSeries::new(
        [(mean_open, 0), (mean_open, max_count)],
        RED.stroke_width(2),
    ))?;

    root.present()?;
    Ok(())
}

/// Figure 6: Pocket Size vs Dynamics (scatter plot)
/// X: Residue Count, Y: CV SASA, color by dynamics category
pub fn generate_pocket_size_vs_dynamics(
    path: &Path,
    sites: &[SiteFigureData],
) -> Result<()> {
    let root = SVGBackend::new(path, (700, 500)).into_drawing_area();
    root.fill(&WHITE)?;

    if sites.is_empty() {
        root.draw(&Text::new("No data", (350, 250), ("sans-serif", 20).into_font().color(&BLACK)))?;
        root.present()?;
        return Ok(());
    }

    let max_res = sites.iter().map(|s| s.residue_count).max().unwrap_or(10) as f64;
    let max_cv = sites.iter().map(|s| s.cv_sasa).fold(0.0f64, f64::max).max(1.0);

    let mut chart = ChartBuilder::on(&root)
        .caption("Pocket Size vs Dynamics", ("sans-serif", 18))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0.0..max_res * 1.1, 0.0..max_cv * 1.1)?;

    chart
        .configure_mesh()
        .x_desc("Residue Count")
        .y_desc("CV SASA")
        .draw()?;

    // Plot points colored by dynamics level
    for site in sites {
        let color = match site.dynamics_level.as_str() {
            "Low" => BLUE,
            "Moderate" => YELLOW,
            "High" => RED,
            _ => BLACK,
        };
        chart.draw_series(std::iter::once(Circle::new(
            (site.residue_count as f64, site.cv_sasa),
            6,
            ShapeStyle::from(&color).filled(),
        )))?;
    }

    root.present()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_volume_vs_time() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("volume.svg");

        let data: Vec<(f32, f64)> = (0..100)
            .map(|i| (i as f32, 200.0 + (i as f64 * 0.1).sin() * 50.0))
            .collect();

        generate_volume_vs_time(&path, &data, "site_001").unwrap();
        assert!(path.exists());
    }

    #[test]
    fn test_heatmap_color() {
        let c0 = heatmap_color(0.0);
        let c1 = heatmap_color(1.0);
        let c5 = heatmap_color(0.5);

        assert_eq!(c0.2, 255); // Blue
        assert_eq!(c1.0, 255); // Red
        assert!(c5.0 > 200 && c5.1 > 200 && c5.2 > 200); // White-ish
    }

    #[test]
    fn test_cryptic_figures_empty() {
        let tmp = TempDir::new().unwrap();
        let sites: Vec<SiteFigureData> = vec![];

        // Should not panic on empty data
        generate_dynamics_distribution(&tmp.path().join("dyn.svg"), &sites).unwrap();
        generate_cryptic_classification(&tmp.path().join("class.svg"), &sites).unwrap();
        generate_cv_sasa_distribution(&tmp.path().join("cv.svg"), &sites).unwrap();
    }
}
