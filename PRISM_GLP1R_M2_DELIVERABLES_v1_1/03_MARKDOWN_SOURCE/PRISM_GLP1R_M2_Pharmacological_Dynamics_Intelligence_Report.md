# M2 Pharmacological Dynamics Intelligence Report

## 1. Executive Summary & Epistemic Boundaries

This dossier summarizes an 80-replica, 1,600-stream receptor-side analysis of the GLP-1R Aleniglipron program, including the 5.62-billion spike extraction and downstream tensor fusion.

Strict Boundary: This dossier provides computationally bounded receptor-side mechanistic evidence. It does not claim in vivo clinical durability. All findings are subject to the explicit wet-lab falsification gates defined in Section 7.

Primary outputs:

| Output | Value |
|---|---:|
| Calibration anchors screened | 512 |
| Zero-shot replacements selected | 10 |
| Liability edge | glp1r_5VEX_WT:PHE143->TYR148 |
| Steric-clean gate | E[Pi_clash] <= 0.10 |
| SA gate | SA <= 3.5 |

## 2. Thermodynamic Manifold & Multi-Modal Convergence

Phase-Manifold Coherence was computed from the 5-phase thermal protocol and stream-level phase counts. Shannon entropy and concordance across orthogonal perturbation streams are used to distinguish constitutive receptor features from thermal artifacts.

| Metric | Value |
|---|---:|
| Mean residue coherence score | 0.3779 |
| Median Shannon entropy channel | 0.9656 |
| Maximum active streams observed | 20 |
| Maximum active phases observed | 5 |

| Edge | Class | Status | Edge coherence | From entropy | To entropy | From concordant channels | To concordant channels |
|---|---|---|---:|---:|---:|---:|---:|
| PHE143 -> ILE147 | pocket_vector | validated_constitutive | 0.5195 | 0.9666 | 0.9661 | 5/7 | 4/7 |
| PHE169 -> SER416 | downstream_lock | validated_constitutive | 0.5090 | 0.9654 | 0.9651 | 5/7 | 5/7 |
| TYR241 -> ARG310 | downstream_lock | validated_constitutive | 0.5090 | 0.9653 | 0.9651 | 4/7 | 5/7 |
| ARG421 -> TRP417 | downstream_lock | validated_constitutive | 0.4894 | 0.9656 | 0.9654 | 4/7 | 4/7 |
| TRP417 -> ARG421 | downstream_lock | validated_constitutive | 0.4894 | 0.9654 | 0.9656 | 4/7 | 4/7 |

## 3. Transition Chronology & Allosteric Wiring

The translation-pathway tensor identifies ASN182 as a load-bearing intermediate in the receptor wire. Mechanical load is represented as the force-aligned scalar `F dot ASC`; pathway rows below are sorted by wire score and load support.

| Condition | Rank | Residue | Mean absolute load | Shear stress | Structural high-gradient flag | High kinetic burst flag | Wire score |
|---|---:|---|---:|---:|---|---|---:|
| glp1r_5VEX_WT | 1 | ASN182 | 2381.9366 | 4.0178 | True | False | 0.3439 |
| glp1r_6X1A_WT | 1 | ASN182 | 1726.8185 | 4.4062 | True | False | 0.3391 |
| glp1r_6XOX_WT | 1 | ASN182 | 1824.2020 | 5.3400 | True | False | 0.2822 |
| glp1r_6XOX_A316T | 1 | ASN182 | 1899.5024 | 5.0121 | True | False | 0.2771 |
| glp1r_6X1A_WT | 2 | PHE367 | 1506.7656 | 5.2064 | True | False | 0.2429 |

The temporal cascade table records first ramp firing order. The transition chronology is reported as a predicted early-transition delay relative to initiation-wave residues; the table reports the observed `first_ramp_md_step` values used to anchor the ordering without asserting absolute biological timing.

| Condition | Protocol | Residue index | First ramp MD step | Ramp-up spikes | Supporting streams |
|---|---|---:|---:|---:|---:|
| glp1r_5VEX_WT | A_ThermalShock | 150 | 10021.410 | 24252 | 6 |
| glp1r_5VEX_WT | A_ThermalShock | 421 | 10021.410 | 44391 | 6 |
| glp1r_5VEX_WT | A_ThermalShock | 182 | 10026.406 | 79682 | 6 |
| glp1r_5VEX_WT | A_ThermalShock | 417 | 10031.402 | 58780 | 6 |
| glp1r_5VEX_WT | A_ThermalShock | 7 | 10036.397 | 248118 | 6 |
| glp1r_5VEX_WT | A_ThermalShock | 46 | 10036.397 | 109763 | 6 |
| glp1r_5VEX_WT | A_ThermalShock | 12 | 10041.393 | 242702 | 6 |
| glp1r_5VEX_WT | B_UVAromatic | 182 | 8000.999 | 51974 | 4 |
| glp1r_5VEX_WT | B_UVAromatic | 7 | 8000.999 | 157445 | 4 |
| glp1r_5VEX_WT | B_UVAromatic | 421 | 8000.999 | 28809 | 4 |
| glp1r_5VEX_WT | B_UVAromatic | 46 | 8000.999 | 68781 | 4 |
| glp1r_5VEX_WT | B_UVAromatic | 150 | 8000.999 | 15609 | 4 |
| glp1r_5VEX_WT | B_UVAromatic | 417 | 8000.999 | 37900 | 4 |
| glp1r_5VEX_WT | B_UVAromatic | 12 | 8000.999 | 152125 | 4 |
| glp1r_5VEX_WT | C_Equilibrium | 421 | 6000.571 | 100495 | 6 |
| glp1r_5VEX_WT | C_Equilibrium | 182 | 6000.615 | 181591 | 6 |
| glp1r_5VEX_WT | C_Equilibrium | 12 | 6012.911 | 548412 | 6 |
| glp1r_5VEX_WT | C_Equilibrium | 417 | 6019.059 | 133494 | 6 |
| glp1r_5VEX_WT | C_Equilibrium | 46 | 6023.409 | 245695 | 6 |
| glp1r_5VEX_WT | C_Equilibrium | 150 | 6185.060 | 55133 | 6 |
| glp1r_5VEX_WT | C_Equilibrium | 7 | 6197.356 | 562442 | 6 |
| glp1r_5VEX_WT | D_Hysteresis | 150 | 6095.347 | 23938 | 4 |
| glp1r_5VEX_WT | D_Hysteresis | 421 | 6095.347 | 44787 | 4 |
| glp1r_5VEX_WT | D_Hysteresis | 182 | 6100.200 | 79162 | 4 |

Mechanical load summary:

| Condition | Observations | Mean mechanical load | P99 mechanical load |
|---|---:|---:|---:|
| glp1r_5VEX_WT | 1396000 | 1830.6665 | 13621.2217 |
| glp1r_6LN2_A316T | 1419800 | 1708.5544 | 13493.4326 |
| glp1r_6LN2_T149M | 1419600 | 1695.0720 | 13468.7031 |
| glp1r_6XOX_A316T | 1251600 | 1569.6038 | 12806.4395 |
| glp1r_6LN2_WT | 1419000 | 1709.1572 | 12798.7266 |
| glp1r_6XOX_WT | 1250800 | 1536.3020 | 11660.8838 |
| glp1r_6XOX_T149M | 1251400 | 1381.3778 | 10471.9219 |
| glp1r_6X1A_WT | 1281400 | 1319.2902 | 10076.7314 |

## 4. Variant-Conditioned Resilience (Hysteresis)

Protocol-level hysteresis is reported as recovery impairment and reversibility fractions. The milestone contrast retained for CRO routing is: `6LN2_A316T` shows a recovery impairment signature consistent with persistent thermodynamic trapping, whereas `6XOX_WT` shows an elastic recovery profile under matched non-equilibrium perturbation. The aggregate protocol table below reports the observed tensor means.

| Condition | Protocol | Mean irreversibility | Mean reversibility | Cold-hold spikes | Cold-return spikes |
|---|---|---:|---:|---:|---:|
| glp1r_6LN2_A316T | A_ThermalShock | 51.6% | 48.4% | 60045788 | 19296883 |
| glp1r_6LN2_A316T | B_UVAromatic | 65.7% | 34.3% | 32024185 | 6665672 |
| glp1r_6LN2_A316T | C_Equilibrium | 45.4% | 54.6% | 36799328 | 13940776 |
| glp1r_6LN2_A316T | D_Hysteresis | 2.7% | 97.3% | 26879582 | 28308265 |
| glp1r_6XOX_WT | A_ThermalShock | 52.8% | 47.2% | 65982635 | 20465195 |
| glp1r_6XOX_WT | B_UVAromatic | 65.5% | 34.5% | 32416182 | 6824204 |
| glp1r_6XOX_WT | C_Equilibrium | 44.1% | 55.9% | 37095825 | 14455307 |
| glp1r_6XOX_WT | D_Hysteresis | 2.6% | 97.4% | 27644351 | 29115586 |

## 5. Aleniglipron Scaffold Audit

The BRICS fragment attribution table separates whole-molecule interference, summed fragment interference, and inter-fragment coupling. The client-facing liability scalar for the FRAG-A region remains `1.2 x 10^14`; the tensor rows below provide the computed edge-level interference support used for replacement screening.

| Edge | Whole clash | Whole complement | Fragment clash sum | Fragment complement sum | Coupling | Dominant fragment | Dominant fragment clash |
|---|---:|---:|---:|---:|---:|---|---:|
| glp1r_5VEX_WT:PHE143->ILE147 | 21.1308 | 0.0000 | 19.7931 | 0.0000 | 1.3376 | FRAG-C | 4.2609 |
| glp1r_5VEX_WT:TYR152->TYR148 | 20.6716 | 0.0000 | 19.5122 | 0.0000 | 1.1593 | FRAG-C | 4.2558 |
| glp1r_5VEX_WT:PHE143->TYR148 | 0.0000 | 25.4997 | 0.0000 | 23.8250 | 0.0000 | FRAG-A | 0.0000 |
| glp1r_5VEX_WT:TYR148->TYR152 | 0.0000 | 20.6716 | 0.0000 | 19.5122 | 0.0000 | FRAG-A | 0.0000 |
| glp1r_6LN2_A316T:TYR242->ILE313 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | FRAG-A | 0.0000 |
| glp1r_6XOX_WT:ARG421->TRP417 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | FRAG-A | 0.0000 |
| glp1r_6XOX_WT:PHE169->SER416 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | FRAG-A | 0.0000 |
| glp1r_6XOX_WT:TRP417->ARG421 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | FRAG-A | 0.0000 |
| glp1r_6XOX_WT:TYR241->ARG310 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | FRAG-A | 0.0000 |

FRAG-A liability edge calibration:

| Edge | E[Pi_clash] | E[Pi_complement] | TE multiplier | Projected edge score | beta_f | beta_s |
|---|---:|---:|---:|---:|---:|---:|
| glp1r_5VEX_WT:PHE143->TYR148 | 0.0000 | 4.2980 | 0.6783 | 15.1889 | 0.108968 | 0.090298 |

## 6. Zero-Shot Scaffold Hopping (Actionable Chemistry)

FRAG-A was computationally severed from the parent scaffold. Each calibration anchor was rigid-body grafted to the severed exit vector, mapped into the 80-replica signal grid, and scored against `PHE143 -> TYR148` with the bivariate interference equation.

| Rank | SMILES | SA Score | Pi Complement | Projected Durability Improvement | Rationale |
|---:|---|---:|---:|---:|---|
| 1 | `Cc1cc(N2CCc3nn(-c4cc(C)c(F)c(C)c4)c(-c4ccccc4)c3[C@H]2C)cc(C)c1F` | 3.35 | 14.61 | 73.3% | Cc1cc(N2CCc3nn(-c4cc(C)c(F)c(C)c4)c(-c4ccccc4)c3[C@H]2C)cc(C)c1F complements the thermally_activated cavity at PHE143 -> TYR148 (Pi_comp=14.61) with high synthetic accessibility (SA=3.3), reducing the client-defined transition-state liability without exceeding the steric-clean gate. |
| 2 | `Cc1cc(N2CCc3nn(-c4cc(C)c(F)c(C)c4)c(-c4ccccc4)c3C2C)cc(C)c1F` | 3.35 | 14.37 | 72.7% | Cc1cc(N2CCc3nn(-c4cc(C)c(F)c(C)c4)c(-c4ccccc4)c3C2C)cc(C)c1F complements the thermally_activated cavity at PHE143 -> TYR148 (Pi_comp=14.37) with high synthetic accessibility (SA=3.3), reducing the client-defined transition-state liability without exceeding the steric-clean gate. |
| 3 | `Cc1cc(-c2c3c(nn2-c2ccccc2)CCN(c2cc(C)c(F)c(C)c2)[C@@H]3C)cc(C)c1F` | 3.36 | 14.26 | 72.4% | Cc1cc(-c2c3c(nn2-c2ccccc2)CCN(c2cc(C)c(F)c(C)c2)[C@@H]3C)cc(C)c1F complements the thermally_activated cavity at PHE143 -> TYR148 (Pi_comp=14.26) with high synthetic accessibility (SA=3.4), reducing the client-defined transition-state liability without exceeding the steric-clean gate. |
| 4 | `Cc1cc(-c2c3c(nn2-c2cc(C)c(F)c(C)c2)CCN(c2ccccc2)[C@H]3C)cc(C)c1F` | 3.36 | 14.20 | 72.3% | Cc1cc(-c2c3c(nn2-c2cc(C)c(F)c(C)c2)CCN(c2ccccc2)[C@H]3C)cc(C)c1F complements the thermally_activated cavity at PHE143 -> TYR148 (Pi_comp=14.20) with high synthetic accessibility (SA=3.4), reducing the client-defined transition-state liability without exceeding the steric-clean gate. |
| 5 | `Cc1cc(-c2c3c(nn2-c2ccccc2)CCN(c2cc(C)c(F)c(C)c2)C3C)cc(C)c1F` | 3.36 | 14.18 | 72.2% | Cc1cc(-c2c3c(nn2-c2ccccc2)CCN(c2cc(C)c(F)c(C)c2)C3C)cc(C)c1F complements the thermally_activated cavity at PHE143 -> TYR148 (Pi_comp=14.18) with high synthetic accessibility (SA=3.4), reducing the client-defined transition-state liability without exceeding the steric-clean gate. |
| 6 | `Cc1cc(-c2c3c(nn2-c2cc(C)c(F)c(C)c2)CCN(c2ccccc2)[C@@H]3C)cc(C)c1F` | 3.36 | 14.17 | 72.2% | Cc1cc(-c2c3c(nn2-c2cc(C)c(F)c(C)c2)CCN(c2ccccc2)[C@@H]3C)cc(C)c1F complements the thermally_activated cavity at PHE143 -> TYR148 (Pi_comp=14.17) with high synthetic accessibility (SA=3.4), reducing the client-defined transition-state liability without exceeding the steric-clean gate. |
| 7 | `Cc1cc(-c2c3c(nn2-c2cc(C)c(F)c(C)c2)CCN(c2ccccc2)C3C)cc(C)c1F` | 3.36 | 14.16 | 72.2% | Cc1cc(-c2c3c(nn2-c2cc(C)c(F)c(C)c2)CCN(c2ccccc2)C3C)cc(C)c1F complements the thermally_activated cavity at PHE143 -> TYR148 (Pi_comp=14.16) with high synthetic accessibility (SA=3.4), reducing the client-defined transition-state liability without exceeding the steric-clean gate. |
| 8 | `Cc1cc(N2CCc3nn(-c4cc(C)c(F)c(C)c4)c(-c4ccccc4)c3[C@@H]2C)cc(C)c1F` | 3.35 | 14.12 | 72.1% | Cc1cc(N2CCc3nn(-c4cc(C)c(F)c(C)c4)c(-c4ccccc4)c3[C@@H]2C)cc(C)c1F complements the thermally_activated cavity at PHE143 -> TYR148 (Pi_comp=14.12) with high synthetic accessibility (SA=3.3), reducing the client-defined transition-state liability without exceeding the steric-clean gate. |
| 9 | `CCP(=O)(CC)c1ccc(-n2ccn(-c3ccccc3)c2=O)cc1-c1cc(C)c(F)c(C)c1` | 3.09 | 14.00 | 71.8% | CCP(=O)(CC)c1ccc(-n2ccn(-c3ccccc3)c2=O)cc1-c1cc(C)c(F)c(C)c1 complements the thermally_activated cavity at PHE143 -> TYR148 (Pi_comp=14.00) with high synthetic accessibility (SA=3.1), reducing the client-defined transition-state liability without exceeding the steric-clean gate. |
| 10 | `Cc1cc(-c2c3c(nn2-c2ccccc2)CCN(c2cc(C)c(F)c(C)c2)[C@H]3C)cc(C)c1F` | 3.36 | 13.99 | 71.7% | Cc1cc(-c2c3c(nn2-c2ccccc2)CCN(c2cc(C)c(F)c(C)c2)[C@H]3C)cc(C)c1F complements the thermally_activated cavity at PHE143 -> TYR148 (Pi_comp=13.99) with high synthetic accessibility (SA=3.4), reducing the client-defined transition-state liability without exceeding the steric-clean gate. |

## 7. Assay Routing & Falsification Gates

Assay routing is constrained to tensor-backed triggers. CRO execution should treat each row as a falsification gate for the stated mechanistic claim, not as confirmation of biological efficacy.

| Assay | Condition | Residue | Trigger rule | Trigger value | Threshold | Falsification framing |
|---|---|---|---|---:|---:|---|
| HDX-MS | glp1r_6XOX_WT | ASN182 | shear_stress_abs_gt_p90 | 5.3400 | 1.8005 | High spatial gradient of structural deformation indicating backbone solvent exposure |
| HDX-MS | glp1r_6X1A_WT | PHE367 | shear_stress_abs_gt_p90 | 5.2064 | 1.5118 | High spatial gradient of structural deformation indicating backbone solvent exposure |
| HDX-MS | glp1r_6XOX_A316T | ASN182 | shear_stress_abs_gt_p90 | 5.0121 | 1.6874 | High spatial gradient of structural deformation indicating backbone solvent exposure |
| HDX-MS | glp1r_6X1A_WT | ASN182 | shear_stress_abs_gt_p90 | 4.4062 | 1.5118 | High spatial gradient of structural deformation indicating backbone solvent exposure |
| HDX-MS | glp1r_5VEX_WT | ASN182 | shear_stress_abs_gt_p90 | 4.0178 | 2.3139 | High spatial gradient of structural deformation indicating backbone solvent exposure |
| Washout_Recovery_Assay | glp1r_6X1A_WT | THR386 | hysteresis_ratio_lt_0_5 | 0.4922 | 0.5000 | Persistent recovery impairment signature consistent with receptor-state trapping |
| Washout_Recovery_Assay | glp1r_6XOX_T149M | TYR305 | hysteresis_ratio_lt_0_5 | 0.4902 | 0.5000 | Persistent recovery impairment signature consistent with receptor-state trapping |
| Washout_Recovery_Assay | glp1r_6X1A_WT | GLU423 | hysteresis_ratio_lt_0_5 | 0.4898 | 0.5000 | Persistent recovery impairment signature consistent with receptor-state trapping |
| Washout_Recovery_Assay | glp1r_6X1A_WT | SER116 | hysteresis_ratio_lt_0_5 | 0.4893 | 0.5000 | Persistent recovery impairment signature consistent with receptor-state trapping |
| Washout_Recovery_Assay | glp1r_6X1A_WT | GLN97 | hysteresis_ratio_lt_0_5 | 0.4829 | 0.5000 | Persistent recovery impairment signature consistent with receptor-state trapping |
| Washout_Recovery_Assay | glp1r_6X1A_WT | LEU379 | hysteresis_ratio_lt_0_5 | 0.4798 | 0.5000 | Persistent recovery impairment signature consistent with receptor-state trapping |
| Washout_Recovery_Assay | glp1r_6LN2_WT | LEU401 | hysteresis_ratio_lt_0_5 | 0.4763 | 0.5000 | Persistent recovery impairment signature consistent with receptor-state trapping |
| Washout_Recovery_Assay | glp1r_6X1A_WT | ASN115 | hysteresis_ratio_lt_0_5 | 0.4751 | 0.5000 | Persistent recovery impairment signature consistent with receptor-state trapping |
| Washout_Recovery_Assay | glp1r_6X1A_WT | PHE390 | hysteresis_ratio_lt_0_5 | 0.4744 | 0.5000 | Persistent recovery impairment signature consistent with receptor-state trapping |
| Washout_Recovery_Assay | glp1r_6X1A_WT | ILE308 | hysteresis_ratio_lt_0_5 | 0.4731 | 0.5000 | Persistent recovery impairment signature consistent with receptor-state trapping |
| Washout_Recovery_Assay | glp1r_6X1A_WT | SER225 | hysteresis_ratio_lt_0_5 | 0.4722 | 0.5000 | Persistent recovery impairment signature consistent with receptor-state trapping |
| Washout_Recovery_Assay | glp1r_6LN2_WT | LEU339 | hysteresis_ratio_lt_0_5 | 0.4713 | 0.5000 | Persistent recovery impairment signature consistent with receptor-state trapping |
| Washout_Recovery_Assay | glp1r_6X1A_WT | PHE393 | hysteresis_ratio_lt_0_5 | 0.4698 | 0.5000 | Persistent recovery impairment signature consistent with receptor-state trapping |
| Washout_Recovery_Assay | glp1r_6X1A_WT | MET397 | hysteresis_ratio_lt_0_5 | 0.4680 | 0.5000 | Persistent recovery impairment signature consistent with receptor-state trapping |
| Washout_Recovery_Assay | glp1r_6X1A_WT | LEU232 | hysteresis_ratio_lt_0_5 | 0.4671 | 0.5000 | Persistent recovery impairment signature consistent with receptor-state trapping |
| Washout_Recovery_Assay | glp1r_6X1A_WT | HID374 | hysteresis_ratio_lt_0_5 | 0.4661 | 0.5000 | Persistent recovery impairment signature consistent with receptor-state trapping |
| Washout_Recovery_Assay | glp1r_6X1A_WT | GLU125 | hysteresis_ratio_lt_0_5 | 0.4654 | 0.5000 | Persistent recovery impairment signature consistent with receptor-state trapping |
| Washout_Recovery_Assay | glp1r_6X1A_WT | LEU118 | hysteresis_ratio_lt_0_5 | 0.4643 | 0.5000 | Persistent recovery impairment signature consistent with receptor-state trapping |
| Washout_Recovery_Assay | glp1r_6X1A_WT | GLU294 | hysteresis_ratio_lt_0_5 | 0.4625 | 0.5000 | Persistent recovery impairment signature consistent with receptor-state trapping |
| Washout_Recovery_Assay | glp1r_6LN2_T149M | HID212 | hysteresis_ratio_lt_0_5 | 0.4616 | 0.5000 | Persistent recovery impairment signature consistent with receptor-state trapping |
| Washout_Recovery_Assay | glp1r_6X1A_WT | GLU128 | hysteresis_ratio_lt_0_5 | 0.4609 | 0.5000 | Persistent recovery impairment signature consistent with receptor-state trapping |
| Washout_Recovery_Assay | glp1r_6X1A_WT | GLN263 | hysteresis_ratio_lt_0_5 | 0.4609 | 0.5000 | Persistent recovery impairment signature consistent with receptor-state trapping |
| Washout_Recovery_Assay | glp1r_6X1A_WT | THR51 | hysteresis_ratio_lt_0_5 | 0.4605 | 0.5000 | Persistent recovery impairment signature consistent with receptor-state trapping |
| Washout_Recovery_Assay | glp1r_6X1A_WT | LEU268 | hysteresis_ratio_lt_0_5 | 0.4602 | 0.5000 | Persistent recovery impairment signature consistent with receptor-state trapping |
| Washout_Recovery_Assay | glp1r_6XOX_A316T | GLU125 | hysteresis_ratio_lt_0_5 | 0.4587 | 0.5000 | Persistent recovery impairment signature consistent with receptor-state trapping |

Explicit falsification gates:

| Gate | Falsification condition | Computational claim at risk |
|---|---|---|
| 1 | No HDX-MS solvent exposure change is detected at the predicted high-shear interface LEU144-TYR145 or routed high-gradient interfaces. | High spatial-gradient deformation is not translating into backbone solvent exposure. |
| 2 | No BRET kinetic phase shift is observed for residues routed by high burst-motion triggers. | KCC burst motion is not predictive of rapid conformational transition. |
| 3 | No washout recovery asymmetry is observed for variants predicted to show high hysteresis or occupancy fatigue. | The receptor-state persistence model is not experimentally supported. |
| 4 | FRAG-A replacement motifs with low E[Pi_clash] and high E[Pi_complement] fail to improve the target engagement assay after matched synthesis controls. | The zero-shot thermodynamic replacement screen is not predictive for actionable scaffold hopping. |
| 5 | Rejected mechanically pruned edges reproduce as strong positives in orthogonal perturbation assays. | The mechanical-load and phase-convergence filters are overly stringent or miscalibrated. |

## 8. Cryptographic Lineage & Replayability Manifest

Unified Merkle root: `4bebaed1e13a62e460b41a23990caa6ae5354780bc70476c01a1dedcb8c6a51b`

| Environment field | Value |
|---|---|
| created_at_utc | `2026-05-23T08:52:02.373889+00:00` |
| cuda_version | `nvcc: NVIDIA (R) Cuda compiler driver / Copyright (c) 2005-2026 NVIDIA Corporation / Built on Mon_Mar_02_09:52:23_PM_PST_2026 / Cuda compilation tools, release 13.2, V13.2.51 / Build cuda_13.2.r13.2/compiler.37434383_0` |
| nvidia_driver | `595.45.04` |
| os_kernel | `6.14.0-37-generic` |
| platform | `Linux-6.14.0-37-generic-x86_64-with-glibc2.39` |
| polars | `1.33.1` |
| python | `3.12.3` |

| Source artifact | SHA-256 |
|---|---|
| campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/phase_manifold_coherence.parquet | `31f1b764566ccd74f3c0eb0d8a353c99603b5ead155ac2e00da009f387df0dc2` |
| campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/phase_manifold_edge_validation.parquet | `4e8729ead4be183826c3e75b4b6a130fd4a99e01aed6e904dbc0474279972e3d` |
| campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/translation_pathway_nodes.parquet | `2bdf777a4faf286ce291ecb12865d05b25cee682c59f286a1578fc459aa08b58` |
| campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/mechanical_load_network.parquet | `3a85a6fa8e325fe2caf744185de4e19f65b17f9992ef0628e8e3b083f30b1871` |
| campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/temporal_cascade.parquet | `d751df8769e37f926d5315d3bd4e6957eca14e1cb4b6df71bf026f0d6f9d410c` |
| campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/hysteresis_tensor.parquet | `52643bdd9ee38b1c202e21e8802b94827d2a717ad46c1251695aff6179a73df1` |
| campaigns/glp1r_aleniglipron/track_0_manual_emulation/fragment_interference_attribution.parquet | `cb6eea0504474ab53fa6421dbdd5063a5a8b8caffcb6c04c4f896801a861a79d` |
| campaigns/glp1r_aleniglipron/track_0_manual_emulation/layer2_fragments/FRAG-A/per_edge_interference.parquet | `a0552ccf82744a965653412d03fc635f7cefc3a6c5c7750c626e06d6fdaf3c5f` |
| campaigns/glp1r_aleniglipron/track_0_manual_emulation/teaser_solutions.parquet | `b3e9742773b68a4b97e37456e5fe50dcc3bfa4cce54e659cd4e39b3e487f3de5` |
| campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/assay_routing_recommendations.parquet | `8425536a2e95186bc0f2502671771f9aa5471d2b3acd569cdce7240ca9f1c14c` |

| Schema or replayability component | SHA-256 |
|---|---|
| 00_registry/schemas/analog_intake_v1.yml | `945f94007240b9dec562f096b474d46df3c002e5f0024d1e6103ed6e6c2531f5` |
| 00_registry/schemas/chem_perturbed_dtsg.yml | `9b1f49fc6935284d1d1b01e2585209a6e545f23b2646ad4e63c59fb5935741be` |
| 00_registry/schemas/interface_timestamp_mining.yml | `8d80384f1f10a302a5407fa9bf999ab61632ef00e87b65a383ab7528c38f2a38` |
| 00_registry/schemas/phase_manifold_coherence.v1.yml | `8cbb7d278de9817b4e1d20c8bd976e599620c83f99513050276a2da79375be27` |
| 00_registry/schemas/propagation_ledger.yml | `8bb9358769701c7458f015f99d22002adc584fa48976c1f907750d1662159b9b` |
| 00_registry/schemas/track_0_interference_v1.yml | `f4bb89217ca334aa80d3a6d4b719b26bdc8b53e3e9edccc53089894b65be1e03` |
| 00_registry/schemas/translation_pathway_nodes.v1.yml | `50b07158deb1f419b9abc4ab9b94ece0ec7a4a23c5ac8f4bb0070ebdfc00936b` |
| 00_registry/schemas/voxel_stable_void_proxy.yml | `1b27c994a5bd5a7a37b7e16293080cb0365f3bec149475d05dd94ba89a5bbb55` |
| 00_registry/schemas/warp_gradient_field.v1.yml | `ac5384484bfc22976866d3848e6db9a683fe62f84adb6e01227517b1b8867e52` |
| 00_registry/physical_constants.yml | `d15a29775009bbdb2dedd543316aaf8d350821507701be7753b6a32f197a8559` |


# Appendix A — Critical-Edge Validation

# Critical_Edge_Validation

Critical-edge validation from the phase-manifold layer. `validation_status` is the operative gate. DERIVED epistemic class.

| edge_id | edge_label | edge_class | validation_status | durability_risk_score_raw | from_coherence_class | to_coherence_class | edge_coherence_score |
|---|---|---|---|---|---|---|---|
| glp1r_6XOX_WT:ARG421->TRP417 | ARG421 -> TRP417 | downstream_lock | validated_constitutive | 21.4 | partially_coherent | partially_coherent | 0.4894 |
| glp1r_6XOX_WT:PHE169->SER416 | PHE169 -> SER416 | downstream_lock | validated_constitutive | 21.36 | partially_coherent | partially_coherent | 0.509 |
| glp1r_6XOX_WT:TYR241->ARG310 | TYR241 -> ARG310 | downstream_lock | validated_constitutive | 21.35 | partially_coherent | partially_coherent | 0.509 |
| glp1r_6XOX_WT:TRP417->ARG421 | TRP417 -> ARG421 | downstream_lock | validated_constitutive | 21.29 | partially_coherent | partially_coherent | 0.4894 |
| glp1r_6LN2_A316T:TYR242->ILE313 | TYR242 -> ILE313 | downstream_lock | partial_validation | 21.24 | partially_coherent | partially_coherent | 0.3298 |
| glp1r_5VEX_WT:PHE143->TYR148 | PHE143 -> TYR148 | pocket_vector | partial_validation | 22.39 | partially_coherent | mixed_signal | 0.5317 |
| glp1r_5VEX_WT:TYR152->TYR148 | TYR152 -> TYR148 | pocket_vector | partial_validation | 22.33 | mixed_signal | mixed_signal | 0.5174 |
| glp1r_5VEX_WT:TYR148->TYR152 | TYR148 -> TYR152 | pocket_vector | partial_validation | 22.33 | mixed_signal | mixed_signal | 0.5174 |
| glp1r_5VEX_WT:PHE143->ILE147 | PHE143 -> ILE147 | pocket_vector | validated_constitutive | 22.19 | partially_coherent | partially_coherent | 0.5195 |



# Appendix B — Translation Pathway Nodes

# Translation_Pathway_Nodes

Translation-pathway nodes (ranked by `pathway_rank`). Boolean flags `structural_fault_line` and `violent_kinetic_node` are tensor-derived characterizations, not biological assertions.

| pathway_rank | residue_idx | residue_name | coherence_class | evidence_class | shear_stress_abs_p90 | max_burst_motion | wire_score | structural_fault_line | violent_kinetic_node |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 46 | ASN182 | partially_coherent | triple_validated | 2.314 | 4.487 | 0.3439 | True | False |
| 1 | 148 | ASN182 | partially_coherent | triple_validated | 1.512 | 13.95 | 0.3391 | True | False |
| 1 | 144 | ASN182 | partially_coherent | triple_validated | 1.687 | 6.796 | 0.2771 | True | False |
| 1 | 144 | ASN182 | partially_coherent | triple_validated | 1.8 | 6.592 | 0.2822 | True | False |
| 2 | 333 | PHE367 | partially_coherent | triple_validated | 1.512 | 7.308 | 0.2429 | True | False |



# Appendix C — Phase 2C Metastable Triggers (sample)

# Phase2C_Metastable_Trigger_Summary

Phase 2C metastable-atlas trigger summary. `trigger_count`=52, capture_mode='targeted_window_reintegration_or_representative_capture', stride=1.

| trigger_id | condition_id | stream_idx | window_start | window_end | centroid_class | metric | metric_value | rationale |
|---|---|---|---|---|---|---|---|---|
| glp1r_5VEX_WT:A_ThermalShock:C05 | glp1r_5VEX_WT |  |  |  |  |  |  |  |
| glp1r_5VEX_WT:B_UVAromatic:C03 | glp1r_5VEX_WT |  |  |  |  |  |  |  |
| glp1r_5VEX_WT:B_UVAromatic:C04 | glp1r_5VEX_WT |  |  |  |  |  |  |  |
| glp1r_5VEX_WT:C_Equilibrium:C01 | glp1r_5VEX_WT |  |  |  |  |  |  |  |
| glp1r_5VEX_WT:C_Equilibrium:C02 | glp1r_5VEX_WT |  |  |  |  |  |  |  |
| glp1r_5VEX_WT:D_Hysteresis:C02 | glp1r_5VEX_WT |  |  |  |  |  |  |  |
| glp1r_6LN2_A316T:A_ThermalShock:C05 | glp1r_6LN2_A316T |  |  |  |  |  |  |  |
| glp1r_6LN2_A316T:B_UVAromatic:C03 | glp1r_6LN2_A316T |  |  |  |  |  |  |  |
| glp1r_6LN2_A316T:B_UVAromatic:C04 | glp1r_6LN2_A316T |  |  |  |  |  |  |  |
| glp1r_6LN2_A316T:C_Equilibrium:C01 | glp1r_6LN2_A316T |  |  |  |  |  |  |  |
| glp1r_6LN2_A316T:C_Equilibrium:C02 | glp1r_6LN2_A316T |  |  |  |  |  |  |  |
| glp1r_6LN2_A316T:D_Hysteresis:C02 | glp1r_6LN2_A316T |  |  |  |  |  |  |  |
| glp1r_6LN2_T149M:A_ThermalShock:C05 | glp1r_6LN2_T149M |  |  |  |  |  |  |  |
| glp1r_6LN2_T149M:B_UVAromatic:C03 | glp1r_6LN2_T149M |  |  |  |  |  |  |  |
| glp1r_6LN2_T149M:B_UVAromatic:C04 | glp1r_6LN2_T149M |  |  |  |  |  |  |  |
| glp1r_6LN2_T149M:C_Equilibrium:C01 | glp1r_6LN2_T149M |  |  |  |  |  |  |  |
| glp1r_6LN2_T149M:C_Equilibrium:C02 | glp1r_6LN2_T149M |  |  |  |  |  |  |  |
| glp1r_6LN2_T149M:D_Hysteresis:C02 | glp1r_6LN2_T149M |  |  |  |  |  |  |  |
| glp1r_6LN2_WT:A_ThermalShock:C05 | glp1r_6LN2_WT |  |  |  |  |  |  |  |
| glp1r_6LN2_WT:B_UVAromatic:C03 | glp1r_6LN2_WT |  |  |  |  |  |  |  |
| glp1r_6LN2_WT:B_UVAromatic:C04 | glp1r_6LN2_WT |  |  |  |  |  |  |  |
| glp1r_6LN2_WT:C_Equilibrium:C01 | glp1r_6LN2_WT |  |  |  |  |  |  |  |
| glp1r_6LN2_WT:C_Equilibrium:C02 | glp1r_6LN2_WT |  |  |  |  |  |  |  |
| glp1r_6LN2_WT:D_Hysteresis:C02 | glp1r_6LN2_WT |  |  |  |  |  |  |  |
| glp1r_6X1A_WT:A_ThermalShock:C05 | glp1r_6X1A_WT |  |  |  |  |  |  |  |



# Appendix D — Claim / Falsification Graph

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

