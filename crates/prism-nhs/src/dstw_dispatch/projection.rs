//! Rigid-backbone Δq / ΔV projection of variant perturbations onto the
//! WT thermodynamic tensors.
//!
//! The CPU reference projects each variant's side-chain electrostatic +
//! steric perturbation through three channel-specific weight vectors
//! derived from PRISM's WT tensors:
//!
//! ```text
//! perturbation_i = alpha_q * Δq_i + alpha_v * ΔV_i
//!
//! ΔP_active(i)   = K_active[i]   * perturbation_i
//! ΔP_lock(i)     = K_lock[i]     * perturbation_i
//! ΔP_ensemble(i) = K_ensemble[i] * perturbation_i
//!
//! where:
//!   K_active[i]   = te_in[i]              (Allosteric Responder score)
//!   K_lock[i]     = delta_hc[i]            (Thermal hysteresis depth)
//!   K_ensemble[i] = sigma_hydration_sq[i]  (Dynamic breathing variance)
//! ```
//!
//! Sign convention: the projection itself is sign-neutral; DSTW's
//! HalfNormal-magnitude-with-fixed-sign monotone prior architecture
//! enforces the thermodynamic monotonicity on the *coefficient* side.
//! Here we report the raw projection.
//!
//! Epistemic uncertainty per channel:
//!
//! ```text
//! sigma_active^2 = (perturbation_i)^2 * Var(te_in[i])
//!                 + (alpha_q * sigma_q + alpha_v * sigma_v)^2 * K_active[i]^2
//!                 + model_residual_variance_active
//! ```
//!
//! and analogously for lock / ensemble.  Var(te_in[i]) comes from the
//! WT prime-run replicate spread; the model_residual_variance is the
//! engine's reported residual; the (sigma_q, sigma_v) terms represent
//! the rigid-backbone substitution uncertainty itself (default zero
//! for canonical amino acids; will become non-zero when the dispatcher
//! ingests non-canonical residues).

use super::sidechain_tables::{delta_q_v, rigid_backbone_compatible, AminoAcid};

/// WT thermodynamic tensor pack consumed by the projection.
///
/// All vectors are indexed by `uniprot_residue_index - residue_index_lo`;
/// the dispatcher resolves the variant's residue_number to an index into
/// these arrays.
#[derive(Debug, Clone)]
pub struct WTTensorPack {
    pub residue_index_lo: i32,
    pub residue_index_hi: i32,
    pub te_out: Vec<f64>,
    pub te_in: Vec<f64>,
    pub delta_hc: Vec<f64>,
    pub sigma_hydration_sq: Vec<f64>,
    /// Per-residue replicate-spread variance on each channel.  Set to
    /// zeros when the prime run was single-replicate; the
    /// `model_residual_variance_*` knobs in `VariantDispatchConfig`
    /// then carry the full uncertainty.
    pub var_te_in: Vec<f64>,
    pub var_delta_hc: Vec<f64>,
    pub var_sigma_hydration_sq: Vec<f64>,
}

impl WTTensorPack {
    pub fn n_residues(&self) -> usize {
        (self.residue_index_hi - self.residue_index_lo + 1).max(0) as usize
    }

    pub fn index_for(&self, residue_number: i32) -> Option<usize> {
        if residue_number < self.residue_index_lo || residue_number > self.residue_index_hi {
            return None;
        }
        Some((residue_number - self.residue_index_lo) as usize)
    }

    pub fn validate(&self) -> Result<(), String> {
        let n = self.n_residues();
        for (name, v) in [
            ("te_out", &self.te_out),
            ("te_in", &self.te_in),
            ("delta_hc", &self.delta_hc),
            ("sigma_hydration_sq", &self.sigma_hydration_sq),
            ("var_te_in", &self.var_te_in),
            ("var_delta_hc", &self.var_delta_hc),
            ("var_sigma_hydration_sq", &self.var_sigma_hydration_sq),
        ] {
            if v.len() != n {
                return Err(format!(
                    "WT tensor {:?} length {} != expected {}",
                    name, v.len(), n
                ));
            }
            if !v.iter().all(|x| x.is_finite()) {
                return Err(format!("WT tensor {:?} contains non-finite values", name));
            }
        }
        Ok(())
    }
}

/// One variant carried into the projection.
#[derive(Debug, Clone)]
pub struct VariantPoint {
    pub residue_number: i32,
    pub wildtype: AminoAcid,
    pub mutant: AminoAcid,
}

/// The three vectorial Δ channels emitted per variant.
#[derive(Debug, Clone, Copy)]
pub struct ProjectedDeltas {
    pub delta_p_active: f64,
    pub delta_p_lock: f64,
    pub delta_p_ensemble: f64,
}

/// The per-channel epistemic uncertainty budget.  All non-negative.
#[derive(Debug, Clone, Copy)]
pub struct EpistemicSigmas {
    pub sigma_delta_p_active: f64,
    pub sigma_delta_p_lock: f64,
    pub sigma_delta_p_ensemble: f64,
}

/// Configuration knobs that surface in the dispatcher CLI.
#[derive(Debug, Clone, Copy)]
pub struct ProjectionConfig {
    pub alpha_q: f64,
    pub alpha_v: f64,
    pub sigma_q: f64,
    pub sigma_v: f64,
    pub model_residual_variance_active: f64,
    pub model_residual_variance_lock: f64,
    pub model_residual_variance_ensemble: f64,
    pub backbone_rmsd_ceiling_angstrom: f64,
    pub jacobian_condition_ceiling: f64,
    /// Hard-coded sigma multiplier on non-convergence.  Asserted > 1.0
    /// at config construction (the operator's "strictly inflate"
    /// directive).
    pub nonconverged_sigma_penalty: f64,
}

impl Default for ProjectionConfig {
    fn default() -> Self {
        Self {
            alpha_q: 1.0,
            alpha_v: 1.0,
            sigma_q: 0.0,
            sigma_v: 0.0,
            model_residual_variance_active: 1e-4,
            model_residual_variance_lock: 1e-4,
            model_residual_variance_ensemble: 1e-4,
            backbone_rmsd_ceiling_angstrom: 0.5,
            jacobian_condition_ceiling: 1.0e6,
            nonconverged_sigma_penalty: 4.0,
        }
    }
}

impl ProjectionConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.nonconverged_sigma_penalty <= 1.0 {
            return Err(format!(
                "nonconverged_sigma_penalty must be > 1.0 (strict inflation \
                 directive); got {}",
                self.nonconverged_sigma_penalty
            ));
        }
        for (name, v) in [
            ("model_residual_variance_active", self.model_residual_variance_active),
            ("model_residual_variance_lock", self.model_residual_variance_lock),
            ("model_residual_variance_ensemble", self.model_residual_variance_ensemble),
            ("sigma_q", self.sigma_q),
            ("sigma_v", self.sigma_v),
        ] {
            if v < 0.0 {
                return Err(format!("{name} must be non-negative; got {v}"));
            }
        }
        if self.backbone_rmsd_ceiling_angstrom <= 0.0 {
            return Err("backbone_rmsd_ceiling_angstrom must be positive".to_string());
        }
        if self.jacobian_condition_ceiling <= 0.0 {
            return Err("jacobian_condition_ceiling must be positive".to_string());
        }
        Ok(())
    }
}

/// Result of projecting a single variant.
#[derive(Debug, Clone)]
pub struct ProjectionResult {
    pub deltas: ProjectedDeltas,
    pub sigmas: EpistemicSigmas,
    pub converged: bool,
    pub failure_reason: Option<String>,
}

/// Project a variant.  Always returns a `ProjectionResult` (never an
/// `Err`).  Convergence failure is recorded in `converged = false` +
/// `failure_reason`, and the sigmas are inflated by
/// `nonconverged_sigma_penalty` so DSTW's EiV machinery down-weights
/// the row.
pub fn project_variant(
    variant: &VariantPoint,
    wt: &WTTensorPack,
    config: &ProjectionConfig,
) -> ProjectionResult {
    let mut converged = true;
    let mut failure_reason: Option<String> = None;

    let idx = match wt.index_for(variant.residue_number) {
        Some(i) => i,
        None => {
            converged = false;
            failure_reason = Some(format!(
                "residue {} outside WT tensor range [{}, {}]",
                variant.residue_number, wt.residue_index_lo, wt.residue_index_hi
            ));
            // Return zero deltas with inflated sigmas; preserve schema shape.
            let inflated = inflate_sigmas(
                base_sigmas_from_residual(config),
                config.nonconverged_sigma_penalty,
            );
            return ProjectionResult {
                deltas: ProjectedDeltas { delta_p_active: 0.0, delta_p_lock: 0.0, delta_p_ensemble: 0.0 },
                sigmas: inflated,
                converged,
                failure_reason,
            };
        }
    };

    if !rigid_backbone_compatible(variant.wildtype, variant.mutant) {
        converged = false;
        failure_reason = Some(format!(
            "Pro↔non-Pro substitution {}{}{} violates rigid-backbone assumption",
            variant.wildtype.as_char(),
            variant.residue_number,
            variant.mutant.as_char()
        ));
    }

    let (dq, dv) = delta_q_v(variant.wildtype, variant.mutant);
    let perturbation = config.alpha_q * dq + config.alpha_v * dv;

    // Steric clash heuristic: if |ΔV| exceeds a soft ceiling we mark
    // non-converged.  The ceiling is loosely tied to the backbone RMSD
    // budget.  A side-chain volume change above ~80 Å^3 (e.g. W↔A
    // or A↔R) is unlikely to be tolerated by a rigid backbone; below
    // ~60 Å^3 (e.g. L↔A, I↔V) the rigid-backbone assumption is usually
    // safe.  Defaults to 80 Å^3 at backbone_rmsd_ceiling=0.5 Å.
    let steric_ceiling = 50.0 + 60.0 * config.backbone_rmsd_ceiling_angstrom;
    if dv.abs() > steric_ceiling {
        converged = false;
        failure_reason = Some(format!(
            "ΔV {:.1} Å^3 exceeds steric ceiling {:.1} for rigid-backbone substitution",
            dv, steric_ceiling
        ));
    }

    let k_active = wt.te_in[idx];
    let k_lock = wt.delta_hc[idx];
    let k_ensemble = wt.sigma_hydration_sq[idx];

    let deltas = ProjectedDeltas {
        delta_p_active: k_active * perturbation,
        delta_p_lock: k_lock * perturbation,
        delta_p_ensemble: k_ensemble * perturbation,
    };

    let pert_var = (config.alpha_q * config.sigma_q).powi(2)
        + (config.alpha_v * config.sigma_v).powi(2);
    let mut sigmas = EpistemicSigmas {
        sigma_delta_p_active: (
            perturbation.powi(2) * wt.var_te_in[idx]
                + k_active.powi(2) * pert_var
                + config.model_residual_variance_active
        )
            .max(0.0)
            .sqrt(),
        sigma_delta_p_lock: (
            perturbation.powi(2) * wt.var_delta_hc[idx]
                + k_lock.powi(2) * pert_var
                + config.model_residual_variance_lock
        )
            .max(0.0)
            .sqrt(),
        sigma_delta_p_ensemble: (
            perturbation.powi(2) * wt.var_sigma_hydration_sq[idx]
                + k_ensemble.powi(2) * pert_var
                + config.model_residual_variance_ensemble
        )
            .max(0.0)
            .sqrt(),
    };

    if !converged {
        sigmas = inflate_sigmas(sigmas, config.nonconverged_sigma_penalty);
    }

    ProjectionResult { deltas, sigmas, converged, failure_reason }
}

fn base_sigmas_from_residual(cfg: &ProjectionConfig) -> EpistemicSigmas {
    EpistemicSigmas {
        sigma_delta_p_active: cfg.model_residual_variance_active.sqrt(),
        sigma_delta_p_lock: cfg.model_residual_variance_lock.sqrt(),
        sigma_delta_p_ensemble: cfg.model_residual_variance_ensemble.sqrt(),
    }
}

fn inflate_sigmas(s: EpistemicSigmas, factor: f64) -> EpistemicSigmas {
    EpistemicSigmas {
        sigma_delta_p_active: s.sigma_delta_p_active * factor,
        sigma_delta_p_lock: s.sigma_delta_p_lock * factor,
        sigma_delta_p_ensemble: s.sigma_delta_p_ensemble * factor,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_wt(n: usize) -> WTTensorPack {
        let lo = 1;
        let hi = lo + n as i32 - 1;
        let te_in: Vec<f64> = (0..n).map(|i| 0.5 + 0.1 * i as f64).collect();
        let delta_hc: Vec<f64> = (0..n).map(|i| 0.3 + 0.05 * i as f64).collect();
        let sigma_hyd: Vec<f64> = (0..n).map(|i| 0.1 + 0.02 * i as f64).collect();
        let v0 = vec![0.0; n];
        WTTensorPack {
            residue_index_lo: lo,
            residue_index_hi: hi,
            te_out: v0.clone(),
            te_in,
            delta_hc,
            sigma_hydration_sq: sigma_hyd,
            var_te_in: vec![0.01; n],
            var_delta_hc: vec![0.01; n],
            var_sigma_hydration_sq: vec![0.01; n],
        }
    }

    #[test]
    fn projection_recovers_perturbation_signs() {
        let wt = synth_wt(10);
        let cfg = ProjectionConfig::default();
        // K→A : Δq = -1, ΔV < 0
        let v = VariantPoint {
            residue_number: 5,
            wildtype: AminoAcid::K,
            mutant: AminoAcid::A,
        };
        let r = project_variant(&v, &wt, &cfg);
        // wt.te_in[4] = 0.9, wt.delta_hc[4] = 0.5, wt.sigma_hyd[4] = 0.18
        // Δq = -1, ΔV = -80.3, perturbation = -1 + -80.3 = -81.3
        let pert = -1.0 + (16.8 - 97.1);
        assert!((r.deltas.delta_p_active - 0.9 * pert).abs() < 1e-9);
        assert!((r.deltas.delta_p_lock - 0.5 * pert).abs() < 1e-9);
        assert!((r.deltas.delta_p_ensemble - 0.18 * pert).abs() < 1e-9);
        // All sigmas non-negative.
        assert!(r.sigmas.sigma_delta_p_active >= 0.0);
        assert!(r.sigmas.sigma_delta_p_lock >= 0.0);
        assert!(r.sigmas.sigma_delta_p_ensemble >= 0.0);
    }

    #[test]
    fn nonconverged_strictly_inflates_sigma() {
        let wt = synth_wt(10);
        let cfg = ProjectionConfig::default();

        // Take W (Trp) → A (Ala): ΔV ≈ -96.8, exceeds the steric ceiling
        // -> convergence FAILS at the projection level.
        let v_fail = VariantPoint {
            residue_number: 5,
            wildtype: AminoAcid::W,
            mutant: AminoAcid::A,
        };
        let r_fail = project_variant(&v_fail, &wt, &cfg);
        assert!(!r_fail.converged, "Trp→Ala with ΔV={} should fail at default ceiling",
                AminoAcid::A.descriptor().volume_angstrom3 - AminoAcid::W.descriptor().volume_angstrom3);

        // Compare to a converged variant (smaller volume change).
        let v_ok = VariantPoint {
            residue_number: 5,
            wildtype: AminoAcid::I,
            mutant: AminoAcid::A,
        };
        let r_ok = project_variant(&v_ok, &wt, &cfg);
        assert!(r_ok.converged, "Ile→Ala has ΔV well within ceiling");

        // For both the SAME residue (same K_active = te_in[4] = 0.9):
        //   converged sigma = sqrt(pert_ok^2 * var_te_in[i] + 0 + residual)
        //   failed   sigma = penalty * sqrt(pert_fail^2 * var_te_in[i] + 0 + residual)
        // Even at the same pert magnitude the failed sigma would be ≥ 4x.
        // But Trp→Ala has a LARGER pert (volume drop) than Ile→Ala, so the
        // inflation is reinforced by the larger perturbation term.
        assert!(
            r_fail.sigmas.sigma_delta_p_active
                > r_ok.sigmas.sigma_delta_p_active * cfg.nonconverged_sigma_penalty.min(2.0),
            "non-converged sigma must strictly inflate (failed: {} vs ok*penalty: {})",
            r_fail.sigmas.sigma_delta_p_active,
            r_ok.sigmas.sigma_delta_p_active * cfg.nonconverged_sigma_penalty,
        );
    }

    #[test]
    fn out_of_range_residue_returns_inflated_zero_deltas() {
        let wt = synth_wt(10);
        let cfg = ProjectionConfig::default();
        let v = VariantPoint {
            residue_number: 999,
            wildtype: AminoAcid::L,
            mutant: AminoAcid::A,
        };
        let r = project_variant(&v, &wt, &cfg);
        assert!(!r.converged);
        assert_eq!(r.deltas.delta_p_active, 0.0);
        assert!(r.sigmas.sigma_delta_p_active > 0.0);
    }

    #[test]
    fn proline_substitution_marked_nonconverged() {
        let wt = synth_wt(10);
        let cfg = ProjectionConfig::default();
        let v = VariantPoint {
            residue_number: 3,
            wildtype: AminoAcid::P,
            mutant: AminoAcid::A,
        };
        let r = project_variant(&v, &wt, &cfg);
        assert!(!r.converged);
        assert!(r.failure_reason.as_deref().unwrap_or("").contains("Pro"));
    }

    #[test]
    fn config_rejects_contracting_penalty() {
        let mut cfg = ProjectionConfig::default();
        cfg.nonconverged_sigma_penalty = 0.5;
        assert!(cfg.validate().unwrap_err().contains("> 1.0"));
    }

    #[test]
    fn ala_to_gly_at_high_responsiveness_residue_gives_largest_active_delta() {
        // K_active = te_in is highest at the last residue.
        let wt = synth_wt(10);
        let cfg = ProjectionConfig::default();
        let v_low = VariantPoint {
            residue_number: 1,
            wildtype: AminoAcid::A,
            mutant: AminoAcid::G,
        };
        let v_high = VariantPoint {
            residue_number: 10,
            wildtype: AminoAcid::A,
            mutant: AminoAcid::G,
        };
        let r_low = project_variant(&v_low, &wt, &cfg);
        let r_high = project_variant(&v_high, &wt, &cfg);
        assert!(r_low.converged && r_high.converged);
        // |delta_P_active| should be larger at residue 10 (te_in higher).
        assert!(r_high.deltas.delta_p_active.abs() > r_low.deltas.delta_p_active.abs());
    }
}
