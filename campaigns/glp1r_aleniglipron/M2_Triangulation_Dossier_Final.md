# M2 Phase 2E Triangulation Dossier

## 1. ZERO-TRUST REPLAYABILITY MANIFEST

Unified Merkle Root: `4bebaed1e13a62e460b41a23990caa6ae5354780bc70476c01a1dedcb8c6a51b`

| Environment field | Value |
|---|---|
| created_at_utc | `2026-05-23T08:52:02.373889+00:00` |
| cuda_version | `nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Mon_Mar_02_09:52:23_PM_PST_2026
Cuda compilation tools, release 13.2, V13.2.51
Build cuda_13.2.r13.2/compiler.37434383_0` |
| nvidia_driver | `595.45.04` |
| os_kernel | `6.14.0-37-generic` |
| platform | `Linux-6.14.0-37-generic-x86_64-with-glibc2.39` |
| polars | `1.33.1` |
| python | `3.12.3` |

| Replayability component | Count |
|---|---:|
| Schema files hashed | 9 |
| Propagation ledgers hashed | 67 |
| Physical constants hash present | True |

| Schema file | SHA-256 |
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

## 2. ASSAY HANDOFF PACKAGE

The assay routing matrix maps tensor-supported residue findings to wet-lab assays using explicit physical triggers and non-figurative terminology.

| Assay | Condition | Residue | Trigger rule | Trigger value | Rationale |
|---|---|---|---|---:|---|
| HDX-MS | glp1r_6XOX_WT | ASN182 | shear_stress_abs_gt_p90 | 5.3400 | High spatial gradient of structural deformation indicating backbone solvent exposure |
| HDX-MS | glp1r_6X1A_WT | PHE367 | shear_stress_abs_gt_p90 | 5.2064 | High spatial gradient of structural deformation indicating backbone solvent exposure |
| HDX-MS | glp1r_6XOX_A316T | ASN182 | shear_stress_abs_gt_p90 | 5.0121 | High spatial gradient of structural deformation indicating backbone solvent exposure |
| HDX-MS | glp1r_6X1A_WT | ASN182 | shear_stress_abs_gt_p90 | 4.4062 | High spatial gradient of structural deformation indicating backbone solvent exposure |
| HDX-MS | glp1r_5VEX_WT | ASN182 | shear_stress_abs_gt_p90 | 4.0178 | High spatial gradient of structural deformation indicating backbone solvent exposure |
| Washout_Recovery_Assay | glp1r_6X1A_WT | THR386 | hysteresis_ratio_lt_0_5 | 0.4922 | Persistent recovery impairment signature consistent with receptor-state trapping |
| Washout_Recovery_Assay | glp1r_6XOX_T149M | TYR305 | hysteresis_ratio_lt_0_5 | 0.4902 | Persistent recovery impairment signature consistent with receptor-state trapping |
| Washout_Recovery_Assay | glp1r_6X1A_WT | GLU423 | hysteresis_ratio_lt_0_5 | 0.4898 | Persistent recovery impairment signature consistent with receptor-state trapping |
| Washout_Recovery_Assay | glp1r_6X1A_WT | SER116 | hysteresis_ratio_lt_0_5 | 0.4893 | Persistent recovery impairment signature consistent with receptor-state trapping |
| Washout_Recovery_Assay | glp1r_6X1A_WT | GLN97 | hysteresis_ratio_lt_0_5 | 0.4829 | Persistent recovery impairment signature consistent with receptor-state trapping |
| Washout_Recovery_Assay | glp1r_6X1A_WT | LEU379 | hysteresis_ratio_lt_0_5 | 0.4798 | Persistent recovery impairment signature consistent with receptor-state trapping |
| Washout_Recovery_Assay | glp1r_6LN2_WT | LEU401 | hysteresis_ratio_lt_0_5 | 0.4763 | Persistent recovery impairment signature consistent with receptor-state trapping |
| Washout_Recovery_Assay | glp1r_6X1A_WT | ASN115 | hysteresis_ratio_lt_0_5 | 0.4751 | Persistent recovery impairment signature consistent with receptor-state trapping |
| Washout_Recovery_Assay | glp1r_6X1A_WT | PHE390 | hysteresis_ratio_lt_0_5 | 0.4744 | Persistent recovery impairment signature consistent with receptor-state trapping |
| Washout_Recovery_Assay | glp1r_6X1A_WT | ILE308 | hysteresis_ratio_lt_0_5 | 0.4731 | Persistent recovery impairment signature consistent with receptor-state trapping |
| Washout_Recovery_Assay | glp1r_6X1A_WT | SER225 | hysteresis_ratio_lt_0_5 | 0.4722 | Persistent recovery impairment signature consistent with receptor-state trapping |
| Washout_Recovery_Assay | glp1r_6LN2_WT | LEU339 | hysteresis_ratio_lt_0_5 | 0.4713 | Persistent recovery impairment signature consistent with receptor-state trapping |
| Washout_Recovery_Assay | glp1r_6X1A_WT | PHE393 | hysteresis_ratio_lt_0_5 | 0.4698 | Persistent recovery impairment signature consistent with receptor-state trapping |
| Washout_Recovery_Assay | glp1r_6X1A_WT | MET397 | hysteresis_ratio_lt_0_5 | 0.4680 | Persistent recovery impairment signature consistent with receptor-state trapping |
| Washout_Recovery_Assay | glp1r_6X1A_WT | LEU232 | hysteresis_ratio_lt_0_5 | 0.4671 | Persistent recovery impairment signature consistent with receptor-state trapping |
| Washout_Recovery_Assay | glp1r_6X1A_WT | HID374 | hysteresis_ratio_lt_0_5 | 0.4661 | Persistent recovery impairment signature consistent with receptor-state trapping |
| Washout_Recovery_Assay | glp1r_6X1A_WT | GLU125 | hysteresis_ratio_lt_0_5 | 0.4654 | Persistent recovery impairment signature consistent with receptor-state trapping |
| Washout_Recovery_Assay | glp1r_6X1A_WT | LEU118 | hysteresis_ratio_lt_0_5 | 0.4643 | Persistent recovery impairment signature consistent with receptor-state trapping |
| Washout_Recovery_Assay | glp1r_6X1A_WT | GLU294 | hysteresis_ratio_lt_0_5 | 0.4625 | Persistent recovery impairment signature consistent with receptor-state trapping |
| Washout_Recovery_Assay | glp1r_6LN2_T149M | HID212 | hysteresis_ratio_lt_0_5 | 0.4616 | Persistent recovery impairment signature consistent with receptor-state trapping |
| Washout_Recovery_Assay | glp1r_6X1A_WT | GLU128 | hysteresis_ratio_lt_0_5 | 0.4609 | Persistent recovery impairment signature consistent with receptor-state trapping |
| Washout_Recovery_Assay | glp1r_6X1A_WT | GLN263 | hysteresis_ratio_lt_0_5 | 0.4609 | Persistent recovery impairment signature consistent with receptor-state trapping |
| Washout_Recovery_Assay | glp1r_6X1A_WT | THR51 | hysteresis_ratio_lt_0_5 | 0.4605 | Persistent recovery impairment signature consistent with receptor-state trapping |
| Washout_Recovery_Assay | glp1r_6X1A_WT | LEU268 | hysteresis_ratio_lt_0_5 | 0.4602 | Persistent recovery impairment signature consistent with receptor-state trapping |
| Washout_Recovery_Assay | glp1r_6XOX_A316T | GLU125 | hysteresis_ratio_lt_0_5 | 0.4587 | Persistent recovery impairment signature consistent with receptor-state trapping |

## 3. PROBABILISTIC TRANSITION CHRONOLOGY

Temporal transition clusters were inferred from `first_ramp_md_step` using a BIC-selected one-dimensional Gaussian mixture model. The table reports cluster confidence and temporal overlap entropy so stochastic boundary uncertainty remains explicit.

| Cluster | Members | Centroid MD step | Cluster confidence | Temporal overlap entropy | Stability metric |
|---:|---:|---:|---:|---:|---:|
| 1 | 2496 | 6015.89 | 0.8849 | 0.2181 | 55.6047 |
| 2 | 3430 | 6158.18 | 0.8465 | 0.1978 | 884.7646 |
| 3 | 2395 | 8001.26 | 0.9826 | 0.0352 | 0.1978 |
| 4 | 568 | 8085.65 | 0.9965 | 0.0000 | 1261.3595 |
| 5 | 2963 | 10015.63 | 0.9802 | 0.0001 | 351.6621 |

## 4. TRI-STATE ALENIGLIPRON FIBER GRAPH

FRAG-A biases the receptor toward persistent lock-state occupancy at Cluster 2.

| Fiber type | Edge | Fragment | Cluster | TE multiplier | Interference load | Occupancy fatigue |
|---|---|---|---:|---:|---:|---:|
| constitutive_fiber | ARG421 -> TRP417 | FRAG-A | 1 | 1.0000 | 0.0000 | 0.0047 |
| constitutive_fiber | PHE169 -> SER416 | FRAG-A | 1 | 1.0000 | 0.0000 | 0.0050 |
| constitutive_fiber | TYR241 -> ARG310 | FRAG-A | 1 | 1.0000 | 0.0000 | 0.0050 |
| constitutive_fiber | TRP417 -> ARG421 | FRAG-A | 1 | 1.0000 | 0.0000 | 0.0047 |
| ligand_amplified_fiber | PHE143 -> ILE147 | FRAG-C | 2 | 10.0000 | 21.1308 | 0.0050 |
| weak_or_unresolved_fiber | TYR152 -> TYR148 | FRAG-C | 2 | 9.5119 | 20.6716 | 0.0050 |
| weak_or_unresolved_fiber | TYR242 -> ILE313 | FRAG-A | 2 | 1.0000 | 0.0000 | 0.0053 |
| weak_or_unresolved_fiber | TYR148 -> TYR152 | FRAG-A | 1 | 0.1546 | 20.6716 | 0.0050 |
| weak_or_unresolved_fiber | PHE143 -> TYR148 | FRAG-A | 2 | 0.1000 | 25.4997 | 0.0049 |

Fiber interpretation:

| Fiber type | Count |
|---|---:|
| constitutive_fiber | 4 |
| ligand_amplified_fiber | 1 |
| weak_or_unresolved_fiber | 4 |

## 5. MEDICINAL CHEMISTRY ACTION MATRIX

The SAR contingency register is variant-aware and epistemic-aware. Each candidate vector is evaluated by thermodynamic ray-casting: `Score(v)=sum(Pi_complement)-sum(Pi_clash)-sum(ShearStress)`. High-risk epistemic expansions are reported as `epistemic_mirage` when atom-edge pose uncertainty exceeds the ray-casting threshold.

### Universal Growth Vectors

**No universal growth vectors met the conserved thermally_activated cavity criterion. All current exit vectors exhibit high epistemic uncertainty ($U_{pose}$) and require scaffold rigidification before expansion.**

### Epistemic Mirage Vectors

| Rule | Vector | Edge | Atom | U_pose | Threshold | Rationale |
|---|---|---|---|---:|---:|---|
| SAR-1222 | Vector 1222 | TYR152 -> TYR148 | C22 | 0.0553 | 0.0251 | Defer expansion at Vector 1222; atom-level pose uncertainty exceeds the ray-casting threshold and requires scaffold rigidification before growth-vector optimization. |
| SAR-1417 | Vector 1417 | TYR148 -> TYR152 | C22 | 0.0553 | 0.0251 | Defer expansion at Vector 1417; atom-level pose uncertainty exceeds the ray-casting threshold and requires scaffold rigidification before growth-vector optimization. |
| SAR-1416 | Vector 1416 | TYR148 -> TYR152 | C22 | 0.0553 | 0.0251 | Defer expansion at Vector 1416; atom-level pose uncertainty exceeds the ray-casting threshold and requires scaffold rigidification before growth-vector optimization. |
| SAR-1415 | Vector 1415 | TYR148 -> TYR152 | C22 | 0.0553 | 0.0251 | Defer expansion at Vector 1415; atom-level pose uncertainty exceeds the ray-casting threshold and requires scaffold rigidification before growth-vector optimization. |
| SAR-1224 | Vector 1224 | TYR152 -> TYR148 | C22 | 0.0553 | 0.0251 | Defer expansion at Vector 1224; atom-level pose uncertainty exceeds the ray-casting threshold and requires scaffold rigidification before growth-vector optimization. |
| SAR-1223 | Vector 1223 | TYR152 -> TYR148 | C22 | 0.0553 | 0.0251 | Defer expansion at Vector 1223; atom-level pose uncertainty exceeds the ray-casting threshold and requires scaffold rigidification before growth-vector optimization. |
| SAR-1422 | Vector 1422 | TYR148 -> TYR152 | F24 | 0.0515 | 0.0251 | Defer expansion at Vector 1422; atom-level pose uncertainty exceeds the ray-casting threshold and requires scaffold rigidification before growth-vector optimization. |
| SAR-1423 | Vector 1423 | TYR148 -> TYR152 | F24 | 0.0515 | 0.0251 | Defer expansion at Vector 1423; atom-level pose uncertainty exceeds the ray-casting threshold and requires scaffold rigidification before growth-vector optimization. |
| SAR-1421 | Vector 1421 | TYR148 -> TYR152 | F24 | 0.0515 | 0.0251 | Defer expansion at Vector 1421; atom-level pose uncertainty exceeds the ray-casting threshold and requires scaffold rigidification before growth-vector optimization. |
| SAR-1230 | Vector 1230 | TYR152 -> TYR148 | F24 | 0.0515 | 0.0251 | Defer expansion at Vector 1230; atom-level pose uncertainty exceeds the ray-casting threshold and requires scaffold rigidification before growth-vector optimization. |
| SAR-1229 | Vector 1229 | TYR152 -> TYR148 | F24 | 0.0515 | 0.0251 | Defer expansion at Vector 1229; atom-level pose uncertainty exceeds the ray-casting threshold and requires scaffold rigidification before growth-vector optimization. |
| SAR-1228 | Vector 1228 | TYR152 -> TYR148 | F24 | 0.0515 | 0.0251 | Defer expansion at Vector 1228; atom-level pose uncertainty exceeds the ray-casting threshold and requires scaffold rigidification before growth-vector optimization. |
| SAR-1655 | Vector 1655 | PHE143 -> ILE147 | C37 | 0.0488 | 0.0251 | Defer expansion at Vector 1655; atom-level pose uncertainty exceeds the ray-casting threshold and requires scaffold rigidification before growth-vector optimization. |
| SAR-1654 | Vector 1654 | PHE143 -> ILE147 | C37 | 0.0488 | 0.0251 | Defer expansion at Vector 1654; atom-level pose uncertainty exceeds the ray-casting threshold and requires scaffold rigidification before growth-vector optimization. |
| SAR-1653 | Vector 1653 | PHE143 -> ILE147 | C37 | 0.0488 | 0.0251 | Defer expansion at Vector 1653; atom-level pose uncertainty exceeds the ray-casting threshold and requires scaffold rigidification before growth-vector optimization. |
| SAR-1678 | Vector 1678 | PHE143 -> ILE147 | C45 | 0.0468 | 0.0251 | Defer expansion at Vector 1678; atom-level pose uncertainty exceeds the ray-casting threshold and requires scaffold rigidification before growth-vector optimization. |
| SAR-1677 | Vector 1677 | PHE143 -> ILE147 | C45 | 0.0468 | 0.0251 | Defer expansion at Vector 1677; atom-level pose uncertainty exceeds the ray-casting threshold and requires scaffold rigidification before growth-vector optimization. |
| SAR-1676 | Vector 1676 | PHE143 -> ILE147 | C45 | 0.0468 | 0.0251 | Defer expansion at Vector 1676; atom-level pose uncertainty exceeds the ray-casting threshold and requires scaffold rigidification before growth-vector optimization. |
| SAR-1708 | Vector 1708 | PHE143 -> ILE147 | C55 | 0.0452 | 0.0251 | Defer expansion at Vector 1708; atom-level pose uncertainty exceeds the ray-casting threshold and requires scaffold rigidification before growth-vector optimization. |
| SAR-1707 | Vector 1707 | PHE143 -> ILE147 | C55 | 0.0452 | 0.0251 | Defer expansion at Vector 1707; atom-level pose uncertainty exceeds the ray-casting threshold and requires scaffold rigidification before growth-vector optimization. |

### Prohibited Rigidification Zones

| Rule | Vector | Edge | Class | Ray score | Clash integral | Shear integral | Rationale |
|---|---|---|---|---:|---:|---:|---|
| SAR-1714 | Vector 1714 | PHE143 -> ILE147 | prohibited_rigidification_zone | -8.4458 | 2.5098 | 5.9360 | Halt rigidifying expansion at Vector 1714; the ray intersects a positive clash field near a receptor lock interface. |
| SAR-1328 | Vector 1328 | TYR152 -> TYR148 | prohibited_rigidification_zone | -8.4458 | 2.5098 | 5.9360 | Halt rigidifying expansion at Vector 1328; the ray intersects a positive clash field near a receptor lock interface. |
| SAR-1727 | Vector 1727 | PHE143 -> ILE147 | prohibited_rigidification_zone | -8.2538 | 2.3853 | 5.8685 | Halt rigidifying expansion at Vector 1727; the ray intersects a positive clash field near a receptor lock interface. |
| SAR-1189 | Vector 1189 | TYR152 -> TYR148 | prohibited_rigidification_zone | -8.2254 | 2.6790 | 5.5464 | Halt rigidifying expansion at Vector 1189; the ray intersects a positive clash field near a receptor lock interface. |
| SAR-1575 | Vector 1575 | PHE143 -> ILE147 | prohibited_rigidification_zone | -8.2254 | 2.6790 | 5.5464 | Halt rigidifying expansion at Vector 1575; the ray intersects a positive clash field near a receptor lock interface. |
| SAR-1623 | Vector 1623 | PHE143 -> ILE147 | prohibited_rigidification_zone | -8.1544 | 2.3719 | 5.7825 | Halt rigidifying expansion at Vector 1623; the ray intersects a positive clash field near a receptor lock interface. |
| SAR-1191 | Vector 1191 | TYR152 -> TYR148 | prohibited_rigidification_zone | -7.9600 | 2.3868 | 5.5732 | Halt rigidifying expansion at Vector 1191; the ray intersects a positive clash field near a receptor lock interface. |
| SAR-1577 | Vector 1577 | PHE143 -> ILE147 | prohibited_rigidification_zone | -7.9600 | 2.3868 | 5.5732 | Halt rigidifying expansion at Vector 1577; the ray intersects a positive clash field near a receptor lock interface. |
| SAR-1200 | Vector 1200 | TYR152 -> TYR148 | prohibited_rigidification_zone | -7.9466 | 2.9252 | 5.0214 | Halt rigidifying expansion at Vector 1200; the ray intersects a positive clash field near a receptor lock interface. |
| SAR-1586 | Vector 1586 | PHE143 -> ILE147 | prohibited_rigidification_zone | -7.9466 | 2.9252 | 5.0214 | Halt rigidifying expansion at Vector 1586; the ray intersects a positive clash field near a receptor lock interface. |
| SAR-1190 | Vector 1190 | TYR152 -> TYR148 | prohibited_rigidification_zone | -7.9340 | 2.6767 | 5.2574 | Halt rigidifying expansion at Vector 1190; the ray intersects a positive clash field near a receptor lock interface. |
| SAR-1576 | Vector 1576 | PHE143 -> ILE147 | prohibited_rigidification_zone | -7.9340 | 2.6767 | 5.2574 | Halt rigidifying expansion at Vector 1576; the ray intersects a positive clash field near a receptor lock interface. |
| SAR-1600 | Vector 1600 | PHE143 -> ILE147 | prohibited_rigidification_zone | -7.9019 | 2.5233 | 5.3786 | Halt rigidifying expansion at Vector 1600; the ray intersects a positive clash field near a receptor lock interface. |
| SAR-1646 | Vector 1646 | PHE143 -> ILE147 | prohibited_rigidification_zone | -7.8535 | 2.4255 | 5.4280 | Halt rigidifying expansion at Vector 1646; the ray intersects a positive clash field near a receptor lock interface. |
| SAR-1260 | Vector 1260 | TYR152 -> TYR148 | prohibited_rigidification_zone | -7.8535 | 2.4255 | 5.4280 | Halt rigidifying expansion at Vector 1260; the ray intersects a positive clash field near a receptor lock interface. |
| SAR-1263 | Vector 1263 | TYR152 -> TYR148 | prohibited_rigidification_zone | -7.7664 | 2.3741 | 5.3923 | Halt rigidifying expansion at Vector 1263; the ray intersects a positive clash field near a receptor lock interface. |
| SAR-1579 | Vector 1579 | PHE143 -> ILE147 | prohibited_rigidification_zone | -7.7480 | 2.4172 | 5.3308 | Halt rigidifying expansion at Vector 1579; the ray intersects a positive clash field near a receptor lock interface. |
| SAR-1193 | Vector 1193 | TYR152 -> TYR148 | prohibited_rigidification_zone | -7.7480 | 2.4172 | 5.3308 | Halt rigidifying expansion at Vector 1193; the ray intersects a positive clash field near a receptor lock interface. |
| SAR-1597 | Vector 1597 | PHE143 -> ILE147 | prohibited_rigidification_zone | -7.6872 | 2.3698 | 5.3173 | Halt rigidifying expansion at Vector 1597; the ray intersects a positive clash field near a receptor lock interface. |
| SAR-1572 | Vector 1572 | PHE143 -> ILE147 | prohibited_rigidification_zone | -7.6299 | 2.4830 | 5.1469 | Halt rigidifying expansion at Vector 1572; the ray intersects a positive clash field near a receptor lock interface. |

## 6. NEGATIVE EVIDENCE REGISTER

The following pathways were rejected due to lack of mechanical load support or failure to survive multi-modal convergence.

| Condition | Edge | Class | Rejection basis | Risk score | Validation status |
|---|---|---|---|---:|---|
| glp1r_6LN2_A316T | 431->435 | downstream_lock | mechanically_pruned | 0.0000 | phase_validation_missing |
| glp1r_5VEX_WT | 408->403 | downstream_lock | mechanically_pruned | 0.0000 | phase_validation_missing |
| glp1r_6LN2_WT | 428->432 | downstream_lock | mechanically_pruned | 0.0000 | phase_validation_missing |
| glp1r_5VEX_WT | 354->351 | downstream_lock | mechanically_pruned | 0.0000 | phase_validation_missing |
| glp1r_5VEX_WT | 376->312 | downstream_lock | mechanically_pruned | 0.0000 | phase_validation_missing |
| glp1r_6LN2_WT | 421->416 | downstream_lock | mechanically_pruned | 0.0000 | phase_validation_missing |
| glp1r_6LN2_A316T | 431->434 | downstream_lock | mechanically_pruned | 0.0000 | phase_validation_missing |
| glp1r_5VEX_WT | 320->317 | downstream_lock | mechanically_pruned | 0.0000 | phase_validation_missing |
| glp1r_6LN2_A316T | 421->417 | downstream_lock | mechanically_pruned | 0.0000 | phase_validation_missing |
| glp1r_5VEX_WT | 319->322 | downstream_lock | mechanically_pruned | 0.0000 | phase_validation_missing |
| glp1r_5VEX_WT | 306->294 | downstream_lock | mechanically_pruned | 0.0000 | phase_validation_missing |
| glp1r_6LN2_T149M | 414->418 | downstream_lock | mechanically_pruned | 0.0000 | phase_validation_missing |
| glp1r_6LN2_T149M | 409->412 | downstream_lock | mechanically_pruned | 0.0000 | phase_validation_missing |
| glp1r_6LN2_T149M | 418->413 | downstream_lock | mechanically_pruned | 0.0000 | phase_validation_missing |
| glp1r_5VEX_WT | 375->317 | downstream_lock | mechanically_pruned | 0.0000 | phase_validation_missing |
| glp1r_5VEX_WT | 420->417 | downstream_lock | mechanically_pruned | 0.0000 | phase_validation_missing |
| glp1r_5VEX_WT | 341->348 | downstream_lock | mechanically_pruned | 0.0000 | phase_validation_missing |
| glp1r_5VEX_WT | 321->316 | downstream_lock | mechanically_pruned | 0.0000 | phase_validation_missing |
| glp1r_6LN2_T149M | 405->409 | downstream_lock | mechanically_pruned | 0.0000 | phase_validation_missing |
| glp1r_6LN2_WT | 407->410 | downstream_lock | mechanically_pruned | 0.0000 | phase_validation_missing |

## 7. EXPLICIT FALSIFICATION GATES

| Gate | Falsification condition | Computational claim at risk |
|---|---|---|
| 1 | No HDX-MS solvent exposure change is detected at predicted high-shear interfaces from the assay routing matrix. | High spatial-gradient deformation is not translating into backbone exposure. |
| 2 | No BRET kinetic phase shift is observed for residues routed by high kinetic burst regions. | KCC burst motion is not predictive of rapid conformational transition. |
| 3 | No recovery asymmetry is observed in washout assays for variants with high occupancy fatigue index. | The bounded fatigue model is not capturing persistent receptor-state occupancy. |
| 4 | Ligand perturbation assays show equivalent response on constitutive and induced fibers. | The tri-state separation between constitutive, ligand-amplified, and induced fibers is not experimentally supported. |
| 5 | Proposed universal growth vectors fail to improve activity or selectivity after scaffold rigidification controls. | The dynamic pharmacophore does not identify productive phase-aware exit vectors. |
| 6 | Edges marked as rejected in the Negative Evidence Register reproduce as strong wet-lab positives. | The mechanical-load and multi-modal convergence filters are overly stringent or miscalibrated. |