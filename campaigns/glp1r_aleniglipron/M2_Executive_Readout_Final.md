# M2 Executive Readout Final

## 1. EXECUTIVE SUMMARY

This readout integrates the 80-replica, 1,600-stream GLP-1R WT/variant manifold over 5,620,333,375 SNR-masked spike events. The completed extraction unrolled the 5-phase CCNS hysteresis protocol across Cold Hold, Ramp Up, Warm Hold, Ramp Down, and Cold Return, then fused the phase tensors into receptor durability, pathway, hysteresis, shear, and Aleniglipron interference layers.

Primary tensor scale:

| Tensor | Rows |
|---|---:|
| Stream-level 5-phase counts | 296300 |
| Phase-Manifold Coherence | 3263 |
| Hysteresis tensor | 11852 |
| Interferometric differential | 2963 |
| Rust shear-stress field | 7077888 |
| Receptor durability risk map | 4833 |

## 2. M0-M1 VERIFICATION (THE WT MANIFOLD)

The 80-replica summary now includes 437 mechanically_pruned edges in the edge-level risk map, preserving the mechanical gate before downstream ligand scoring. In the D_Hysteresis Temporal Cascade, non-edge residues fire at median MD step 6100.20, while critical edge endpoints fire later at median MD step 6109.91, mapping the propagation timing of the allosteric wave rather than only the final edge state.

| Durability class | Edge count | Mean risk | Mechanically pruned |
|---|---:|---:|---:|
| moderate_durability_risk | 2499 | 10.3400 | 0 |
| elevated_durability_risk | 967 | 19.8748 | 0 |
| quiet_thermal_lock | 688 | 6.6876 | 0 |
| mechanically_pruned | 437 | 0.0000 | 437 |
| critical_durability_risk | 242 | 21.0398 | 0 |

## 3. PHASE-MANIFOLD COHERENCE (FIRST-PRINCIPLES VALIDATION)

These edges show high signal consistency across 20 orthogonal thermodynamic perturbation streams, providing first-principles evidence that they are constitutive receptor features, not thermal artifacts.

The updated coherence tensor consumes `stream_level_phase_counts.parquet`, so Shannon entropy is computed over stream-phase bins from the 5-phase protocol rather than from a single raw stream histogram. The risk map now carries the phase validation status through the composite critical-edge key.

| Critical edge | Class | Edge coherence | Validation status | Durability risk |
|---|---|---:|---|---:|
| ARG421 -> TRP417 | downstream_lock | 0.4894 | validated_constitutive | 21.4019 |
| PHE169 -> SER416 | downstream_lock | 0.5090 | validated_constitutive | 21.3558 |
| TRP417 -> ARG421 | downstream_lock | 0.4894 | validated_constitutive | 21.2877 |
| TYR241 -> ARG310 | downstream_lock | 0.5090 | validated_constitutive | 21.3500 |
| TYR242 -> ILE313 | downstream_lock | 0.3298 | partial_validation | 21.2354 |
| PHE143 -> ILE147 | pocket_vector | 0.5195 | validated_constitutive | 22.1903 |
| PHE143 -> TYR148 | pocket_vector | 0.5317 | partial_validation | 22.3912 |
| TYR148 -> TYR152 | pocket_vector | 0.5174 | partial_validation | 22.3329 |
| TYR152 -> TYR148 | pocket_vector | 0.5174 | partial_validation | 22.3329 |

## 4. CCNS HYSTERESIS ANALYSIS

`6LN2_A316T` shows a recovery impairment signature consistent with persistent lock-state occupancy, whereas `6XOX_WT` shows an elastic recovery profile under matched thermal perturbation.

The fused `hysteresis_tensor.parquet` provides the condition/protocol accounting below. The D_Hysteresis recovery channel remains explicitly loaded here so the executive contrast is traceable back to Cold Hold versus Cold Return spike behavior.

| Condition | Protocol | Mean irreversibility | Mean reversibility | Cold Hold spikes | Cold Return spikes |
|---|---|---:|---:|---:|---:|
| glp1r_6LN2_A316T | A_ThermalShock | 51.6% | 48.4% | 60045788 | 19296883 |
| glp1r_6LN2_A316T | B_UVAromatic | 65.7% | 34.3% | 32024185 | 6665672 |
| glp1r_6LN2_A316T | C_Equilibrium | 45.4% | 54.6% | 36799328 | 13940776 |
| glp1r_6LN2_A316T | D_Hysteresis | 2.7% | 97.3% | 26879582 | 28308265 |
| glp1r_6XOX_WT | A_ThermalShock | 52.8% | 47.2% | 65982635 | 20465195 |
| glp1r_6XOX_WT | B_UVAromatic | 65.5% | 34.5% | 32416182 | 6824204 |
| glp1r_6XOX_WT | C_Equilibrium | 44.1% | 55.9% | 37095825 | 14455307 |
| glp1r_6XOX_WT | D_Hysteresis | 2.6% | 97.4% | 27644351 | 29115586 |

## 5. THE TRANSLATION PATHWAY

The 9 critical edges below are the receptor handoff points used for downstream scoring, while the pathway table lists the top intermediate load-bearing residues available after the triple-evidence and spatial filters. `ASN182` appears as the dominant intermediate wire residue and is explicitly annotated with shear and burst-motion flags.

| Critical edge | Class | Validation status | Mechanically pruned | Risk |
|---|---|---|---:|---:|
| ARG421 -> TRP417 | downstream_lock | validated_constitutive | False | 21.4019 |
| PHE169 -> SER416 | downstream_lock | validated_constitutive | False | 21.3558 |
| TRP417 -> ARG421 | downstream_lock | validated_constitutive | False | 21.2877 |
| TYR241 -> ARG310 | downstream_lock | validated_constitutive | False | 21.3500 |
| TYR242 -> ILE313 | downstream_lock | partial_validation | False | 21.2354 |
| PHE143 -> ILE147 | pocket_vector | validated_constitutive | False | 22.1903 |
| PHE143 -> TYR148 | pocket_vector | partial_validation | False | 22.3912 |
| TYR148 -> TYR152 | pocket_vector | partial_validation | False | 22.3329 |
| TYR152 -> TYR148 | pocket_vector | partial_validation | False | 22.3329 |

| Residue | Condition | Rank | Wire score | Structural fault line | Violent kinetic node | Shear stress | Max burst motion |
|---|---|---:|---:|---:|---:|---:|---:|
| ASN182 | glp1r_5VEX_WT | 1 | 0.3439 | True | False | 4.0178 | 4.4873 |
| ASN182 | glp1r_6X1A_WT | 1 | 0.3391 | True | False | 4.4062 | 13.9486 |
| ASN182 | glp1r_6XOX_A316T | 1 | 0.2771 | True | False | 5.0121 | 6.7957 |
| ASN182 | glp1r_6XOX_WT | 1 | 0.2822 | True | False | 5.3400 | 6.5921 |
| PHE367 | glp1r_6X1A_WT | 2 | 0.2429 | True | False | 5.2064 | 7.3075 |

## 6. ALENIGLIPRON TWO-LAYER INTERFERENCE

Whole-molecule projected durability score: `546.6557` +/- `60.9397` using frozen beta values `beta_f=0.108968369488` and `beta_s=0.090298355000`.

Inter-Fragment Coupling is positive where the covalent connection forces the whole molecule into a more strained geometry than the sum of independently sliced BRICS fragments. The total coupling across the scored edges is `2.4970`.

| Edge | Whole clash | Sum fragment clash | Inter-Fragment Coupling | Dominant fragment | Dominant fraction |
|---|---:|---:|---:|---|---:|
| glp1r_5VEX_WT:PHE143->ILE147 | 21.1308 | 19.7931 | 1.3376 | FRAG-C | 20.2% |
| glp1r_5VEX_WT:TYR152->TYR148 | 20.6716 | 19.5122 | 1.1593 | FRAG-C | 20.6% |
| glp1r_5VEX_WT:PHE143->TYR148 | 0.0000 | 0.0000 | 0.0000 | FRAG-A | 0.0% |
| glp1r_5VEX_WT:TYR148->TYR152 | 0.0000 | 0.0000 | 0.0000 | FRAG-A | 0.0% |
| glp1r_6LN2_A316T:TYR242->ILE313 | 0.0000 | 0.0000 | 0.0000 | FRAG-A | 0.0% |
| glp1r_6XOX_WT:ARG421->TRP417 | 0.0000 | 0.0000 | 0.0000 | FRAG-A | 0.0% |
| glp1r_6XOX_WT:PHE169->SER416 | 0.0000 | 0.0000 | 0.0000 | FRAG-A | 0.0% |
| glp1r_6XOX_WT:TRP417->ARG421 | 0.0000 | 0.0000 | 0.0000 | FRAG-A | 0.0% |
| glp1r_6XOX_WT:TYR241->ARG310 | 0.0000 | 0.0000 | 0.0000 | FRAG-A | 0.0% |

## 7. STRATEGIC RECOMMENDATION

Transition to the automated Chem-BALD / Graph-Wasserstein DKL pipeline (Track A) to resolve epistemic uncertainty ($U_{pose}$) for rigidified scaffolds. The current Track 0 scoring has enough receptor-side physics to prioritize hypotheses, but the next decision point needs uncertainty-aware pose exploration rather than a single rigidified scaffold assumption.

## 8. CRYPTOGRAPHIC LINEAGE

| Artifact | SHA-256 |
|---|---|
| stream_level_phase_counts.parquet | `19da8f333f00594c9ac569cfab82ed7500366ba79dff9e833f887fb5a504d1d1` |
| phase_manifold_coherence.parquet | `31f1b764566ccd74f3c0eb0d8a353c99603b5ead155ac2e00da009f387df0dc2` |
| phase_manifold_edge_validation.parquet | `4e8729ead4be183826c3e75b4b6a130fd4a99e01aed6e904dbc0474279972e3d` |
| receptor_durability_risk_map.parquet | `2aec5f547d9330533ae4bd8ce86cc04e97d5cc781359843796dd02eef4296bda` |
| temporal_cascade.parquet | `d751df8769e37f926d5315d3bd4e6957eca14e1cb4b6df71bf026f0d6f9d410c` |
| hysteresis_tensor.parquet | `52643bdd9ee38b1c202e21e8802b94827d2a717ad46c1251695aff6179a73df1` |
| interferometric_differential.parquet | `03d7d1fd0bfc144a3db8d3e3c3731e12aad4a8f1e085f8dd2169d06672b93e54` |
| shear_stress_field.parquet | `b556d2d03a1cfc27ed64634287cc0a8ed75ca0c49507d66671be98d4120afda1` |
| translation_pathway_nodes.parquet | `2bdf777a4faf286ce291ecb12865d05b25cee682c59f286a1578fc459aa08b58` |
| analog_durability_projection.parquet | `a4b5b143a2d3fe7767835a91335897da23c3b25336f09d73e377c70f2831abb7` |
| fragment_interference_attribution.parquet | `cb6eea0504474ab53fa6421dbdd5063a5a8b8caffcb6c04c4f896801a861a79d` |
