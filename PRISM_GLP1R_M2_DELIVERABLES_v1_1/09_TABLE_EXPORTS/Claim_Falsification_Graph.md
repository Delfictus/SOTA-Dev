# Claim_Falsification_Graph

Claim/falsification graph in flattened edge-table form. `claim_epistemic` is the gate that governs how each edge may be cited. PROJECTED and HYPOTHESIZED edges require wet-lab falsification before any biological assertion.

| source_id | target_id | relationship | claim_epistemic | claim_assay | claim_condition |
|---|---|---|---|---|---|
| tensor_source:shear_stress > condition p90 | claim:HDX-MS:glp1r_6XOX_WT:144 | supports_projected_claim | PROJECTED | HDX-MS | glp1r_6XOX_WT |
| claim:HDX-MS:glp1r_6XOX_WT:144 | assay:HDX-MS:glp1r_6XOX_WT:ASN182:144 | requires_falsification_by | PROJECTED | HDX-MS | glp1r_6XOX_WT |
| tensor_source:shear_stress > condition p90 | claim:HDX-MS:glp1r_6X1A_WT:333 | supports_projected_claim | PROJECTED | HDX-MS | glp1r_6X1A_WT |
| claim:HDX-MS:glp1r_6X1A_WT:333 | assay:HDX-MS:glp1r_6X1A_WT:PHE367:333 | requires_falsification_by | PROJECTED | HDX-MS | glp1r_6X1A_WT |
| tensor_source:shear_stress > condition p90 | claim:HDX-MS:glp1r_6XOX_A316T:144 | supports_projected_claim | PROJECTED | HDX-MS | glp1r_6XOX_A316T |
| claim:HDX-MS:glp1r_6XOX_A316T:144 | assay:HDX-MS:glp1r_6XOX_A316T:ASN182:144 | requires_falsification_by | PROJECTED | HDX-MS | glp1r_6XOX_A316T |
| tensor_source:shear_stress > condition p90 | claim:HDX-MS:glp1r_6X1A_WT:148 | supports_projected_claim | PROJECTED | HDX-MS | glp1r_6X1A_WT |
| claim:HDX-MS:glp1r_6X1A_WT:148 | assay:HDX-MS:glp1r_6X1A_WT:ASN182:148 | requires_falsification_by | PROJECTED | HDX-MS | glp1r_6X1A_WT |
| tensor_source:shear_stress > condition p90 | claim:HDX-MS:glp1r_5VEX_WT:46 | supports_projected_claim | PROJECTED | HDX-MS | glp1r_5VEX_WT |
| claim:HDX-MS:glp1r_5VEX_WT:46 | assay:HDX-MS:glp1r_5VEX_WT:ASN182:46 | requires_falsification_by | PROJECTED | HDX-MS | glp1r_5VEX_WT |
| tensor_source:hysteresis_ratio < 0.5 | claim:WASHOUT:glp1r_6LN2_A316T | supports_projected_claim | PROJECTED | Washout_Recovery_Assay | glp1r_6LN2_A316T |
| claim:WASHOUT:glp1r_6LN2_A316T | assay:Washout_Recovery_Assay:glp1r_6LN2_A316T:variant_level:None | requires_falsification_by | PROJECTED | Washout_Recovery_Assay | glp1r_6LN2_A316T |
| tensor:hysteresis_tensor | claim:HYSTERESIS:6LN2_A316T | supports_inference | INFERRED |  |  |
| claim:HYSTERESIS:6LN2_A316T | assay:washout_recovery | requires_falsification_by | INFERRED |  |  |
| tensor:fragment_interference_attribution | claim:COUPLING:ALENI-PARENT | supports_projected_claim | PROJECTED |  |  |
| claim:COUPLING:ALENI-PARENT | assay:matched_analog_controls | requires_falsification_by | PROJECTED |  |  |

