//! PyMOL and ChimeraX session generation

use crate::sites::CrypticSite;
use crate::{dependency_install_instructions, find_chimerax, find_pymol};
use anyhow::{bail, Context, Result};
use std::fs;
use std::path::Path;
use std::process::Command;

/// Generate PyMOL .pse session for a site
pub fn generate_pymol_session(
    site: &CrypticSite,
    pdb_path: &Path,
    site_pdb_path: &Path,
    output_pse: &Path,
    holo_pdb: Option<&Path>,
) -> Result<()> {
    let pymol_path = find_pymol().ok_or_else(|| {
        anyhow::anyhow!(
            "PyMOL not found.\n{}",
            dependency_install_instructions("pymol")
        )
    })?;

    // Create .pml script
    let pml_script = output_pse.with_extension("pml");
    let pml_content = generate_pymol_script(site, pdb_path, site_pdb_path, holo_pdb, output_pse)?;
    fs::write(&pml_script, &pml_content)?;

    // Run PyMOL headless to generate .pse
    let status = Command::new(&pymol_path)
        .args(["-cq", pml_script.to_str().unwrap()])
        .status()
        .with_context(|| format!("Failed to run PyMOL: {}", pymol_path.display()))?;

    if !status.success() {
        bail!("PyMOL exited with error. Check {} for issues.", pml_script.display());
    }

    // Verify output
    if !output_pse.exists() {
        bail!("PyMOL did not generate expected output: {}", output_pse.display());
    }

    // Clean up script (optional - keep for debugging)
    // fs::remove_file(&pml_script)?;

    Ok(())
}

fn generate_pymol_script(
    site: &CrypticSite,
    pdb_path: &Path,
    site_pdb_path: &Path,
    holo_pdb: Option<&Path>,
    output_pse: &Path,
) -> Result<String> {
    let mut script = String::new();

    // ==========================================================================
    // PRISM4D INDUSTRY-STANDARD PYMOL SESSION
    // Matches Schrödinger/MOE/Discovery Studio quality
    // ==========================================================================

    // Header
    script.push_str("# ============================================================================\n");
    script.push_str("# PRISM-4D Cryptic Binding Site Visualization\n");
    script.push_str("# Publication-Quality PyMOL Session\n");
    script.push_str("# ============================================================================\n");
    script.push_str(&format!("# Site: {}\n", site.site_id));
    script.push_str(&format!("# Rank: {} | Confidence: {:.2}\n", site.rank, site.confidence));
    script.push_str(&format!("# Volume: {:.0} A^3 | Druggable: {}\n",
        site.metrics.geometry.volume_mean,
        if site.is_druggable { "YES" } else { "NO" }
    ));
    script.push_str(&format!(
        "# Centroid: [{:.2}, {:.2}, {:.2}]\n\n",
        site.centroid[0], site.centroid[1], site.centroid[2]
    ));

    // ---------------------------------------------------------------------
    // PRISM-4D Brand Color Definitions
    // ---------------------------------------------------------------------
    script.push_str("# PRISM-4D Color Palette\n");
    script.push_str("set_color prism_blue, [0.290, 0.435, 0.647]\n");      // #4A6FA5
    script.push_str("set_color prism_gold, [0.769, 0.639, 0.353]\n");      // #C4A35A
    script.push_str("set_color prism_teal, [0.357, 0.604, 0.545]\n");      // #5B9A8B
    script.push_str("set_color prism_coral, [0.831, 0.451, 0.369]\n");     // #D4735E
    script.push_str("set_color prism_gray, [0.420, 0.447, 0.502]\n");      // #6B7280
    script.push_str("set_color prism_helix, [0.290, 0.435, 0.647]\n");     // Blue for helices
    script.push_str("set_color prism_sheet, [0.769, 0.639, 0.353]\n");     // Gold for sheets
    script.push_str("set_color prism_loop, [0.800, 0.800, 0.800]\n");      // Light gray for loops
    script.push_str("\n");

    // ---------------------------------------------------------------------
    // Load Structures
    // ---------------------------------------------------------------------
    script.push_str("# Load structures\n");
    script.push_str(&format!(
        "load {}, protein\n",
        pdb_path.to_str().unwrap()
    ));
    script.push_str(&format!(
        "load {}, site_atoms\n",
        site_pdb_path.to_str().unwrap()
    ));

    if let Some(holo) = holo_pdb {
        script.push_str(&format!("load {}, holo\n", holo.to_str().unwrap()));
        script.push_str("align holo, protein\n");
    }

    // Residue selection
    let residue_sel = site
        .residues
        .iter()
        .map(|r| r.to_string())
        .collect::<Vec<_>>()
        .join("+");

    script.push_str("\n# Define selections\n");
    script.push_str(&format!(
        "select site_residues, protein and resi {}\n",
        residue_sel
    ));
    script.push_str("select site_sidechains, site_residues and not name C+N+O+CA\n");
    script.push_str("select site_backbone, site_residues and name C+N+O+CA\n");

    // ---------------------------------------------------------------------
    // Professional Styling
    // ---------------------------------------------------------------------
    script.push_str("\n# ============================================================================\n");
    script.push_str("# PROFESSIONAL STYLING\n");
    script.push_str("# ============================================================================\n");

    // Background
    script.push_str("\n# Clean white background\n");
    script.push_str("bg_color white\n");
    script.push_str("set opaque_background, 1\n");

    // Hide everything first
    script.push_str("\n# Reset display\n");
    script.push_str("hide all\n");

    // Cartoon representation with secondary structure coloring
    script.push_str("\n# Cartoon with secondary structure coloring\n");
    script.push_str("show cartoon, protein\n");
    script.push_str("set cartoon_fancy_helices, 1\n");
    script.push_str("set cartoon_fancy_sheets, 1\n");
    script.push_str("set cartoon_smooth_loops, 1\n");
    script.push_str("set cartoon_loop_radius, 0.25\n");
    script.push_str("set cartoon_tube_radius, 0.8\n");
    script.push_str("color prism_loop, protein\n");
    script.push_str("color prism_helix, ss h\n");
    script.push_str("color prism_sheet, ss s\n");

    // Site residues - sticks with element coloring
    script.push_str("\n# Site residues - sticks with professional styling\n");
    script.push_str("show sticks, site_residues\n");
    script.push_str("set stick_radius, 0.15\n");
    script.push_str("set stick_ball, 0\n");
    script.push_str("util.cbaw site_residues\n");  // Color by atom type (C white)
    script.push_str("color prism_teal, site_residues and elem C\n");  // Teal carbons

    // Surface for site
    script.push_str("\n# Semi-transparent surface for binding site\n");
    script.push_str("show surface, site_residues\n");
    script.push_str("set surface_quality, 1\n");
    script.push_str("set solvent_radius, 1.4\n");
    script.push_str("color prism_teal, site_residues\n");
    script.push_str("set transparency, 0.65, site_residues\n");
    script.push_str("set surface_color, prism_teal, site_residues\n");

    // Holo ligand if present
    if holo_pdb.is_some() {
        script.push_str("\n# Reference ligand (holo structure)\n");
        script.push_str("show sticks, holo and organic\n");
        script.push_str("set stick_radius, 0.2, holo and organic\n");
        script.push_str("util.cbag holo and organic\n");  // Color by atom (C green)
        script.push_str("show spheres, holo and organic\n");
        script.push_str("set sphere_scale, 0.2, holo and organic\n");
    }

    // Centroid marker
    script.push_str("\n# Site centroid marker (pocket center)\n");
    script.push_str(&format!(
        "pseudoatom centroid, pos=[{:.2}, {:.2}, {:.2}], label=\"{}\"\n",
        site.centroid[0], site.centroid[1], site.centroid[2], site.site_id
    ));
    script.push_str("show spheres, centroid\n");
    script.push_str("color prism_coral, centroid\n");
    script.push_str("set sphere_scale, 0.6, centroid\n");
    script.push_str("set sphere_transparency, 0.3, centroid\n");

    // Residue labels
    script.push_str("\n# Residue labels (one-letter codes)\n");
    script.push_str("set label_font_id, 7\n");  // Helvetica Bold
    script.push_str("set label_size, 14\n");
    script.push_str("set label_color, black\n");
    script.push_str("set label_outline_color, white\n");
    script.push_str("set label_position, [0, 0, 3]\n");
    script.push_str("label site_residues and name CA, resn+resi\n");

    // ---------------------------------------------------------------------
    // Professional Lighting & Rendering
    // ---------------------------------------------------------------------
    script.push_str("\n# ============================================================================\n");
    script.push_str("# PROFESSIONAL LIGHTING & RENDERING\n");
    script.push_str("# ============================================================================\n");

    script.push_str("\n# Multi-light setup for depth perception\n");
    script.push_str("set light_count, 4\n");
    script.push_str("set spec_reflect, 0.3\n");
    script.push_str("set spec_power, 200\n");
    script.push_str("set spec_direct, 0\n");
    script.push_str("set ambient, 0.25\n");
    script.push_str("set direct, 0.6\n");
    script.push_str("set reflect, 0.4\n");

    script.push_str("\n# Ray tracing settings for publication quality\n");
    script.push_str("set ray_trace_mode, 1\n");
    script.push_str("set ray_trace_gain, 0.1\n");
    script.push_str("set ray_shadows, 1\n");
    script.push_str("set ray_shadow_decay_factor, 0.1\n");
    script.push_str("set ray_shadow_decay_range, 2\n");
    script.push_str("set antialias, 2\n");
    script.push_str("set hash_max, 300\n");

    script.push_str("\n# Depth cueing (fog) for 3D perception\n");
    script.push_str("set depth_cue, 1\n");
    script.push_str("set fog_start, 0.45\n");
    script.push_str("set fog, 0.35\n");

    // ---------------------------------------------------------------------
    // Camera Setup
    // ---------------------------------------------------------------------
    script.push_str("\n# ============================================================================\n");
    script.push_str("# CAMERA SETUP\n");
    script.push_str("# ============================================================================\n");
    script.push_str("center site_residues\n");
    script.push_str("orient site_residues\n");
    script.push_str("zoom site_residues, 8\n");
    script.push_str("turn y, 20\n");  // Slight rotation for better view
    script.push_str("turn x, 10\n");

    // ---------------------------------------------------------------------
    // Deselect and Save
    // ---------------------------------------------------------------------
    script.push_str("\n# Clean up selections and save\n");
    script.push_str("deselect\n");
    script.push_str(&format!(
        "save {}\n",
        output_pse.to_str().unwrap()
    ));

    // Generate PNG render
    let png_path = output_pse.with_extension("png");
    script.push_str(&format!(
        "\n# Render high-resolution image (2400x1800)\n\
        ray 2400, 1800\n\
        png {}, dpi=300\n",
        png_path.to_str().unwrap()
    ));

    script.push_str("\nquit\n");

    Ok(script)
}

/// Generate ChimeraX .cxs session for a site
pub fn generate_chimerax_session(
    site: &CrypticSite,
    pdb_path: &Path,
    site_pdb_path: &Path,
    output_cxs: &Path,
    holo_pdb: Option<&Path>,
) -> Result<()> {
    let chimerax_path = find_chimerax().ok_or_else(|| {
        anyhow::anyhow!(
            "ChimeraX not found.\n{}",
            dependency_install_instructions("chimerax")
        )
    })?;

    // Create .cxc script
    let cxc_script = output_cxs.with_extension("cxc");
    let cxc_content = generate_chimerax_script(site, pdb_path, site_pdb_path, holo_pdb, output_cxs)?;
    fs::write(&cxc_script, &cxc_content)?;

    // Run ChimeraX headless
    let status = Command::new(&chimerax_path)
        .args(["--nogui", "--cmd", &format!("open {}", cxc_script.to_str().unwrap())])
        .status()
        .with_context(|| format!("Failed to run ChimeraX: {}", chimerax_path.display()))?;

    if !status.success() {
        // ChimeraX may exit non-zero but still generate output
        log::warn!("ChimeraX exited with non-zero status");
    }

    // Verify output
    if !output_cxs.exists() {
        bail!("ChimeraX did not generate expected output: {}", output_cxs.display());
    }

    Ok(())
}

fn generate_chimerax_script(
    site: &CrypticSite,
    pdb_path: &Path,
    site_pdb_path: &Path,
    holo_pdb: Option<&Path>,
    output_cxs: &Path,
) -> Result<String> {
    let mut script = String::new();

    // ==========================================================================
    // PRISM4D INDUSTRY-STANDARD CHIMERAX SESSION
    // Matches Schrödinger/MOE/Discovery Studio quality
    // ==========================================================================

    // Header
    script.push_str("# ============================================================================\n");
    script.push_str("# PRISM-4D Cryptic Binding Site Visualization\n");
    script.push_str("# Publication-Quality ChimeraX Session\n");
    script.push_str("# ============================================================================\n");
    script.push_str(&format!("# Site: {}\n", site.site_id));
    script.push_str(&format!("# Rank: {} | Confidence: {:.2}\n", site.rank, site.confidence));
    script.push_str(&format!("# Volume: {:.0} A^3 | Druggable: {}\n",
        site.metrics.geometry.volume_mean,
        if site.is_druggable { "YES" } else { "NO" }
    ));
    script.push_str(&format!(
        "# Centroid: [{:.2}, {:.2}, {:.2}]\n\n",
        site.centroid[0], site.centroid[1], site.centroid[2]
    ));

    // ---------------------------------------------------------------------
    // Load Structures
    // ---------------------------------------------------------------------
    script.push_str("# Load structures\n");
    script.push_str(&format!("open {}\n", pdb_path.to_str().unwrap()));
    script.push_str(&format!("open {}\n", site_pdb_path.to_str().unwrap()));

    if let Some(holo) = holo_pdb {
        script.push_str(&format!("open {}\n", holo.to_str().unwrap()));
        script.push_str("matchmaker #3 to #1\n");
    }

    // Residue selection
    let residue_sel = site
        .residues
        .iter()
        .map(|r| r.to_string())
        .collect::<Vec<_>>()
        .join(",");

    // ---------------------------------------------------------------------
    // Professional Styling
    // ---------------------------------------------------------------------
    script.push_str("\n# ============================================================================\n");
    script.push_str("# PROFESSIONAL STYLING\n");
    script.push_str("# ============================================================================\n");

    // Background
    script.push_str("\n# Clean white background\n");
    script.push_str("set bgColor white\n");

    // Hide everything first, then show cartoon
    script.push_str("\n# Reset display\n");
    script.push_str("hide atoms\n");
    script.push_str("hide ribbons\n");
    script.push_str("cartoon style protein modeHelix tube modeSheet default arrows true\n");
    script.push_str("show cartoons #1\n");

    // Secondary structure coloring (PRISM-4D palette)
    script.push_str("\n# Secondary structure coloring (PRISM-4D palette)\n");
    script.push_str("color #1 #4A6FA5 target c  ; # Blue for helices\n");
    script.push_str("color #1:helix #4A6FA5\n");
    script.push_str("color #1:strand #C4A35A  ; # Gold for sheets\n");
    script.push_str("color #1:coil #CCCCCC    ; # Gray for loops\n");

    // Site residues - atoms with element coloring
    script.push_str("\n# Site residues - sticks with element colors\n");
    script.push_str(&format!("show #1:{} atoms\n", residue_sel));
    script.push_str(&format!("style #1:{} stick\n", residue_sel));
    script.push_str(&format!("color byhetero #1:{}\n", residue_sel));
    script.push_str(&format!("color #1:{} & C #5B9A8B  ; # Teal carbons for site\n", residue_sel));

    // Semi-transparent surface for binding site
    script.push_str("\n# Semi-transparent surface for binding site\n");
    script.push_str(&format!("surface #1:{} enclose #1:{}\n", residue_sel, residue_sel));
    script.push_str(&format!("color #1:{} surfaces #5B9A8B  ; # Teal surface\n", residue_sel));
    script.push_str(&format!("transparency #1:{} 65 surfaces\n", residue_sel));

    // Holo ligand if present
    if holo_pdb.is_some() {
        script.push_str("\n# Reference ligand (holo structure)\n");
        script.push_str("show #3 ligand atoms\n");
        script.push_str("style #3 ligand ball\n");
        script.push_str("color byhetero #3 ligand\n");
        script.push_str("color #3 ligand & C forest green\n");
    }

    // Centroid marker
    script.push_str("\n# Site centroid marker (pocket center)\n");
    script.push_str(&format!(
        "shape sphere center {:.2},{:.2},{:.2} radius 1.2 color #D4735E  ; # Coral sphere\n",
        site.centroid[0], site.centroid[1], site.centroid[2]
    ));
    script.push_str("transparency last 30\n");

    // Residue labels
    script.push_str("\n# Residue labels\n");
    script.push_str(&format!("label #1:{} residues text \"{{0.name}}{{0.number}}\"\n", residue_sel));
    script.push_str("label style #1 height 0.8 color black bgColor white outline true\n");

    // ---------------------------------------------------------------------
    // Professional Lighting & Rendering
    // ---------------------------------------------------------------------
    script.push_str("\n# ============================================================================\n");
    script.push_str("# PROFESSIONAL LIGHTING & RENDERING\n");
    script.push_str("# ============================================================================\n");

    script.push_str("\n# High-quality lighting\n");
    script.push_str("lighting soft\n");
    script.push_str("lighting shadows true\n");
    script.push_str("lighting depthCue true\n");
    script.push_str("lighting depthCueStart 0.5\n");
    script.push_str("lighting depthCueEnd 1.0\n");

    script.push_str("\n# Material settings for better depth\n");
    script.push_str("material dull\n");
    script.push_str("set silhouettes true\n");
    script.push_str("set silhouetteWidth 1.5\n");

    // ---------------------------------------------------------------------
    // Camera Setup
    // ---------------------------------------------------------------------
    script.push_str("\n# ============================================================================\n");
    script.push_str("# CAMERA SETUP\n");
    script.push_str("# ============================================================================\n");
    script.push_str(&format!("view #1:{}\n", residue_sel));
    script.push_str("turn y 20  ; # Slight rotation for better perspective\n");
    script.push_str("turn x 10\n");
    script.push_str("zoom 0.9\n");

    // ---------------------------------------------------------------------
    // Save Session and Render
    // ---------------------------------------------------------------------
    script.push_str("\n# ============================================================================\n");
    script.push_str("# SAVE SESSION AND RENDER\n");
    script.push_str("# ============================================================================\n");

    script.push_str(&format!("save {} format session\n", output_cxs.to_str().unwrap()));

    // Generate PNG render
    let png_path = output_cxs.with_extension("png");
    script.push_str(&format!(
        "\n# Render high-resolution image (2400x1800, 300 DPI equivalent)\n\
        save {} width 2400 height 1800 supersample 4\n",
        png_path.to_str().unwrap()
    ));

    script.push_str("\nexit\n");

    Ok(script)
}

/// Check if session generation is available
pub struct SessionAvailability {
    pub pymol: Option<std::path::PathBuf>,
    pub chimerax: Option<std::path::PathBuf>,
}

impl SessionAvailability {
    pub fn check() -> Self {
        Self {
            pymol: find_pymol(),
            chimerax: find_chimerax(),
        }
    }

    pub fn pymol_available(&self) -> bool {
        self.pymol.is_some()
    }

    pub fn chimerax_available(&self) -> bool {
        self.chimerax.is_some()
    }

    pub fn report_missing(&self) -> Vec<String> {
        let mut missing = Vec::new();
        if self.pymol.is_none() {
            missing.push(dependency_install_instructions("pymol"));
        }
        if self.chimerax.is_none() {
            missing.push(dependency_install_instructions("chimerax"));
        }
        missing
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sites::{
        ChemistryMetrics, GeometryMetrics, PersistenceMetrics, SiteMetrics, UvResponseMetrics,
    };

    fn make_test_site() -> CrypticSite {
        CrypticSite {
            site_id: "site_001".to_string(),
            rank: 1,
            centroid: [10.0, 20.0, 30.0],
            residues: vec![100, 101, 102, 103],
            residue_names: vec![
                "LEU".to_string(),
                "ILE".to_string(),
                "VAL".to_string(),
                "PHE".to_string(),
            ],
            chain_id: "A".to_string(),
            metrics: SiteMetrics {
                persistence: PersistenceMetrics {
                    present_fraction: 0.5,
                    mean_lifetime_frames: 100.0,
                    replica_agreement: 0.8,
                },
                geometry: GeometryMetrics {
                    volume_mean: 300.0,
                    volume_p50: 280.0,
                    volume_p95: 400.0,
                    volume_min: 180.0,
                    volume_max: 420.0,
                    volume_std: 60.0,
                    breathing_amplitude: 240.0,
                    aspect_ratio: Some(1.5),
                    sphericity: Some(0.67),
                    depth_proxy_pocket_a: None,
                    depth_proxy_surface_a: None,
                    mouth_area_proxy_a2: None,
                    mouth_area_total_a2: None,
                    n_openings: None,
                },
                chemistry: ChemistryMetrics {
                    hydrophobic_fraction: 0.75,
                    donor_count: 4,
                    acceptor_count: 3,
                    aromatic_fraction: 0.25,
                    charged_fraction: 0.0,
                },
                uv_response: UvResponseMetrics::default(),
            },
            rank_score: 0.8,
            confidence: 0.85,
            is_druggable: true,
            first_frame: 0,
            last_frame: 1000,
            representative_frame: 500,
        }
    }

    #[test]
    fn test_pymol_script_generation() {
        let site = make_test_site();
        let script = generate_pymol_script(
            &site,
            Path::new("/tmp/protein.pdb"),
            Path::new("/tmp/site.pdb"),
            None,
            Path::new("/tmp/session.pse"),
        )
        .unwrap();

        assert!(script.contains("site_001"));
        assert!(script.contains("100+101+102+103"));
        assert!(script.contains("save /tmp/session.pse"));
    }

    #[test]
    fn test_chimerax_script_generation() {
        let site = make_test_site();
        let script = generate_chimerax_script(
            &site,
            Path::new("/tmp/protein.pdb"),
            Path::new("/tmp/site.pdb"),
            None,
            Path::new("/tmp/session.cxs"),
        )
        .unwrap();

        assert!(script.contains("site_001"));
        assert!(script.contains("100,101,102,103"));
        assert!(script.contains("save /tmp/session.cxs"));
    }
}
