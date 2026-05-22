# PRISM-DSTW Workspace Purge Manifest

Generated: 2026-05-21

This manifest records legacy implementation violations found by `scripts/ci/ban_check.py` after the amended workspace standard was encoded.
Exemptions are line-bound in `00_registry/ban_exemptions.yml`; new violations in an exempted file/rule are not automatically suppressed.
The active GLP-1R Arrow/Polars lineage producers and core CI/provenance primitives are excluded from temporary exemption and must stay clean.

Total contaminated files tracked: 314
Total violation instances tracked: 5688

## Files

### `scripts/admet_predict.py`
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 121
- `print_banned_production_logging`: 11 instance(s), lines 212, 213, 214, 243, 244, 245, 209, 222, 234, 236, 227

### `scripts/analyze_ensemble_pockets.py`
- `bare_except_banned`: 1 instance(s), lines 81
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 341
- `print_banned_production_logging`: 30 instance(s), lines 58, 90, 220, 223, 224, 225, 229, 233, 237, 278, 279, 280, 283, 342, 97, 117, 216, 293, 296, 307, 308, 309, 310, 316, 332, ... (+5 more)

### `scripts/analyze_with_alignment.py`
- `banned_determinant`: 1 instance(s), lines 63
- `broad_exception_banned`: 2 instance(s), lines 304, 795
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 731
- `print_banned_production_logging`: 99 instance(s), lines 223, 376, 377, 378, 379, 388, 389, 390, 391, 398, 399, 416, 430, 431, 432, 437, 438, 450, 451, 452, 453, 454, 459, 460, 461, ... (+74 more)

### `scripts/anchor_point_map.py`
- `broad_exception_banned`: 1 instance(s), lines 499
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 551
- `print_banned_production_logging`: 3 instance(s), lines 552, 555, 560

### `scripts/arrow_to_parquet.py`
- `broad_exception_banned`: 3 instance(s), lines 42, 64, 77
- `direct_parquet_writer_banned`: 1 instance(s), lines 56
- `print_banned_production_logging`: 31 instance(s), lines 35, 36, 48, 49, 70, 71, 72, 88, 96, 133, 134, 135, 136, 137, 138, 25, 52, 86, 92, 93, 94, 126, 130, 131, 144, ... (+6 more)

### `scripts/auto_parameterize_ligands.py`
- `broad_exception_banned`: 1 instance(s), lines 58
- `print_banned_production_logging`: 8 instance(s), lines 26, 97, 98, 89, 32, 56, 59, 60

### `scripts/bayesian_rerank.py`
- `banned_text_table_io`: 1 instance(s), lines 35
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 348
- `print_banned_production_logging`: 21 instance(s), lines 222, 223, 230, 250, 251, 305, 306, 307, 308, 317, 318, 339, 340, 351, 352, 368, 321, 355, 228, 323, 325

### `scripts/benchmark_compare.py`
- `banned_text_table_io`: 1 instance(s), lines 22
- `broad_exception_banned`: 1 instance(s), lines 223
- `print_banned_production_logging`: 3 instance(s), lines 392, 393, 395

### `scripts/benchmark_comparison.py`
- `banned_determinant`: 1 instance(s), lines 220
- `banned_text_table_io`: 1 instance(s), lines 21
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 835
- `print_banned_production_logging`: 69 instance(s), lines 358, 362, 425, 493, 680, 681, 682, 683, 684, 687, 690, 691, 749, 750, 751, 752, 755, 775, 776, 777, 800, 801, 802, 803, 816, ... (+44 more)

### `scripts/blind_benchmark_viral.py`
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 447
- `print_banned_production_logging`: 52 instance(s), lines 311, 312, 313, 314, 397, 398, 399, 400, 401, 420, 421, 422, 423, 457, 458, 134, 170, 320, 321, 322, 323, 333, 334, 341, 345, ... (+27 more)

### `scripts/build_pdf.py`
- `print_banned_production_logging`: 14 instance(s), lines 189, 201, 202, 207, 208, 209, 212, 164, 168, 193, 198, 215, 217, 166

### `scripts/cluster_objective_analysis.py`
- `banned_text_table_io`: 1 instance(s), lines 5
- `broad_exception_banned`: 1 instance(s), lines 723
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 733
- `print_banned_production_logging`: 15 instance(s), lines 735, 736, 737, 738, 739, 740, 741, 742, 743, 745, 753, 754, 719, 747, 724

### `scripts/cluster_to_pocket_mapper.py`
- `banned_text_table_io`: 1 instance(s), lines 5
- `broad_exception_banned`: 1 instance(s), lines 712
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 722
- `print_banned_production_logging`: 24 instance(s), lines 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 745, 746, 709, 713

### `scripts/cluster_void_center_mapper.py`
- `banned_text_table_io`: 1 instance(s), lines 5
- `broad_exception_banned`: 1 instance(s), lines 846
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 856
- `print_banned_production_logging`: 25 instance(s), lines 858, 859, 860, 861, 862, 863, 864, 865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 877, 878, 880, 881, 843, 847

### `scripts/cluster_voxel_void_mapper.py`
- `banned_text_table_io`: 1 instance(s), lines 5
- `broad_exception_banned`: 1 instance(s), lines 829
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 839
- `print_banned_production_logging`: 25 instance(s), lines 841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 857, 858, 859, 860, 861, 863, 864, 826, 830

### `scripts/coherence_prt70_panel_sweep.py`
- `banned_text_table_io`: 1 instance(s), lines 5
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 682
- `print_banned_production_logging`: 8 instance(s), lines 695, 709, 719, 720, 721, 595, 697, 712

### `scripts/coherence_recluster.py`
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 652
- `print_banned_production_logging`: 15 instance(s), lines 308, 543, 549, 594, 606, 702, 723, 557, 597, 609, 704, 709, 713, 719, 563

### `scripts/coherence_sweep.py`
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 599
- `print_banned_production_logging`: 4 instance(s), lines 601, 615, 558, 604

### `scripts/combine_chain_topologies.py`
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 139
- `print_banned_production_logging`: 4 instance(s), lines 142, 40, 131, 149

### `scripts/consensus.py`
- `json_dump_banned_for_analytical_data`: 3 instance(s), lines 548, 570, 596
- `print_banned_production_logging`: 14 instance(s), lines 533, 599, 600, 601, 602, 603, 604, 606, 619, 431, 437, 610, 428, 618

### `scripts/contact_reorg_gate.py`
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 320
- `print_banned_production_logging`: 3 instance(s), lines 321, 325, 328

### `scripts/coupled_spikes_to_parquet.py`
- `direct_parquet_writer_banned`: 3 instance(s), lines 56, 83, 76
- `print_banned_production_logging`: 9 instance(s), lines 104, 108, 109, 110, 111, 90, 98, 101, 80

### `scripts/create_prism_topology.py`
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 79
- `print_banned_production_logging`: 8 instance(s), lines 9, 71, 72, 73, 74, 75, 76, 81

### `scripts/design_brief_builder.py`
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 125
- `print_banned_production_logging`: 1 instance(s), lines 465

### `scripts/docking_prep.py`
- `print_banned_production_logging`: 18 instance(s), lines 346, 347, 348, 349, 396, 331, 339, 354, 365, 369, 373, 388, 393, 394, 360, 375, 381, 383

### `scripts/dossier_full.py`
- `banned_dataframe_engine`: 1 instance(s), lines 15
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 791
- `print_banned_production_logging`: 50 instance(s), lines 55, 56, 57, 58, 129, 130, 131, 132, 133, 134, 135, 157, 216, 592, 593, 594, 705, 706, 707, 725, 726, 727, 730, 793, 794, ... (+25 more)
- `sklearn_model_import_requires_exemption`: 1 instance(s), lines 561

### `scripts/dossier_unified.py`
- `banned_dataframe_engine`: 1 instance(s), lines 16
- `banned_determinant`: 1 instance(s), lines 504
- `broad_exception_banned`: 2 instance(s), lines 476, 1066
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 1395
- `print_banned_production_logging`: 106 instance(s), lines 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 170, 171, 172, 173, 174, 186, 187, 323, 324, 325, 372, 373, 375, 377, 378, ... (+81 more)
- `sklearn_model_import_requires_exemption`: 1 instance(s), lines 787

### `scripts/download_and_setup.py`
- `broad_exception_banned`: 1 instance(s), lines 88
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 252
- `print_banned_production_logging`: 40 instance(s), lines 82, 97, 138, 141, 146, 150, 154, 158, 162, 169, 193, 204, 212, 220, 226, 237, 247, 253, 256, 257, 258, 259, 260, 261, 262, ... (+15 more)

### `scripts/download_atlas.py`
- `broad_exception_banned`: 1 instance(s), lines 178
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 223
- `print_banned_production_logging`: 28 instance(s), lines 151, 152, 153, 154, 155, 156, 157, 158, 161, 182, 183, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 218, ... (+3 more)

### `scripts/download_atlas_v2.py`
- `broad_exception_banned`: 1 instance(s), lines 208
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 265
- `print_banned_production_logging`: 31 instance(s), lines 179, 180, 181, 182, 183, 184, 185, 186, 187, 190, 212, 213, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, ... (+6 more)

### `scripts/download_biosecurity_dataset.py`
- `bare_except_banned`: 1 instance(s), lines 76
- `broad_exception_banned`: 2 instance(s), lines 64, 62
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 218
- `print_banned_production_logging`: 51 instance(s), lines 82, 83, 84, 85, 96, 97, 105, 109, 119, 120, 143, 144, 145, 146, 147, 148, 152, 155, 156, 186, 190, 191, 201, 202, 203, ... (+26 more)

### `scripts/download_expansion_structures.py`
- `broad_exception_banned`: 1 instance(s), lines 128
- `print_banned_production_logging`: 36 instance(s), lines 163, 164, 165, 166, 167, 168, 169, 170, 183, 184, 189, 199, 200, 211, 212, 213, 219, 220, 221, 222, 223, 224, 76, 82, 134, ... (+11 more)

### `scripts/egnn_pocket_ranker.py`
- `print_banned_production_logging`: 14 instance(s), lines 379, 389, 464, 467, 468, 495, 553, 554, 555, 382, 493, 557, 449, 595

### `scripts/egnn_pocket_ranker_v2.py`
- `print_banned_production_logging`: 31 instance(s), lines 873, 881, 894, 895, 896, 956, 957, 961, 962, 963, 1024, 1027, 1028, 1029, 1051, 1052, 1067, 1068, 1141, 1142, 1143, 403, 410, 763, 877, ... (+6 more)

### `scripts/evaluate_benchmark.py`
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 722
- `print_banned_production_logging`: 65 instance(s), lines 314, 315, 316, 317, 318, 319, 321, 323, 492, 493, 494, 497, 503, 504, 505, 517, 521, 522, 525, 541, 542, 548, 549, 550, 590, ... (+40 more)

### `scripts/explicit_solvent/pocket_refinement.py`
- `broad_exception_banned`: 1 instance(s), lines 183

### `scripts/explicit_solvent/refine_from_prism.py`
- `broad_exception_banned`: 2 instance(s), lines 205, 539
- `json_dump_banned_for_analytical_data`: 2 instance(s), lines 315, 366
- `print_banned_production_logging`: 67 instance(s), lines 135, 144, 147, 154, 164, 175, 176, 177, 184, 190, 193, 213, 216, 219, 224, 271, 317, 318, 319, 320, 321, 322, 323, 324, 340, ... (+42 more)

### `scripts/explicit_solvent/water_map_analysis.py`
- `sklearn_model_import_requires_exemption`: 1 instance(s), lines 250

### `scripts/fep/analyze_fep.py`
- `broad_exception_banned`: 1 instance(s), lines 350
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 356
- `print_banned_production_logging`: 2 instance(s), lines 391, 397

### `scripts/fep/prepare_abfe.py`
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 181
- `print_banned_production_logging`: 3 instance(s), lines 291, 292, 293

### `scripts/fep/prepare_rbfe.py`
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 122

### `scripts/fep/prism_to_openfe.py`
- `json_dump_banned_for_analytical_data`: 2 instance(s), lines 127, 273

### `scripts/fep/run_fep.py`
- `broad_exception_banned`: 1 instance(s), lines 343
- `json_dump_banned_for_analytical_data`: 2 instance(s), lines 287, 105
- `print_banned_production_logging`: 4 instance(s), lines 395, 397, 374, 376

### `scripts/filters/filter_pipeline.py`
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 227

### `scripts/filters/stage1_validity.py`
- `broad_exception_banned`: 1 instance(s), lines 42

### `scripts/filters/stage5_novelty.py`
- `broad_exception_banned`: 1 instance(s), lines 59

### `scripts/gating_stack.py`
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 284
- `print_banned_production_logging`: 8 instance(s), lines 285, 287, 288, 292, 303, 327, 307, 319

### `scripts/generate_chimerax_viz.py`
- `banned_dataframe_engine`: 1 instance(s), lines 17
- `print_banned_production_logging`: 20 instance(s), lines 246, 247, 248, 251, 262, 264, 267, 269, 271, 272, 273, 274, 275, 276, 277, 278, 279, 255, 257, 258

### `scripts/generate_comprehensive_figures.py`
- `broad_exception_banned`: 1 instance(s), lines 1501
- `print_banned_production_logging`: 56 instance(s), lines 173, 274, 383, 445, 541, 864, 938, 1038, 1093, 1149, 1313, 1473, 1492, 1493, 1494, 1497, 1506, 1508, 1511, 1514, 1517, 1520, 1526, 1527, 1528, ... (+31 more)

### `scripts/generate_figures.py`
- `print_banned_production_logging`: 9 instance(s), lines 95, 130, 219, 228, 288, 297, 303, 121, 99

### `scripts/generate_md_rmsf.py`
- `banned_text_table_io`: 1 instance(s), lines 17
- `broad_exception_banned`: 2 instance(s), lines 101, 162
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 277
- `print_banned_production_logging`: 28 instance(s), lines 220, 221, 222, 223, 224, 225, 226, 229, 283, 284, 285, 286, 287, 288, 289, 38, 215, 228, 235, 279, 280, 293, 294, 295, 253, ... (+3 more)

### `scripts/generate_publication_data.py`
- `banned_dataframe_engine`: 1 instance(s), lines 17
- `banned_determinant`: 1 instance(s), lines 154
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 816
- `print_banned_production_logging`: 47 instance(s), lines 224, 284, 311, 369, 412, 473, 514, 557, 607, 636, 709, 710, 761, 762, 818, 1034, 1043, 1044, 1045, 1068, 1069, 1072, 1080, 1088, 1093, ... (+22 more)

### `scripts/generate_pymol_viz.py`
- `banned_dataframe_engine`: 1 instance(s), lines 20
- `print_banned_production_logging`: 21 instance(s), lines 373, 374, 375, 378, 389, 391, 394, 396, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 382, 384, 385

### `scripts/generate_rmsf_from_bfactors.py`
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 257
- `print_banned_production_logging`: 23 instance(s), lines 172, 173, 174, 175, 176, 177, 178, 179, 259, 260, 261, 262, 263, 264, 265, 266, 253, 271, 272, 273, 285, 201, 209

### `scripts/generate_v4_report.py`
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 713
- `print_banned_production_logging`: 15 instance(s), lines 706, 711, 715, 719, 723, 726, 727, 728, 729, 730, 731, 732, 744, 703, 742

### `scripts/genphore/generate.py`
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 85
- `print_banned_production_logging`: 1 instance(s), lines 257

### `scripts/genphore/run_pgmg.py`
- `broad_exception_banned`: 1 instance(s), lines 116

### `scripts/genphore/run_phoregen.py`
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 169

### `scripts/glycan_preprocessor.py`
- `bare_except_banned`: 1 instance(s), lines 443
- `print_banned_production_logging`: 19 instance(s), lines 239, 240, 243, 267, 321, 322, 494, 499, 503, 510, 511, 242, 361, 362, 488, 490, 386, 392, 401

### `scripts/gpu_dock.py`
- `broad_exception_banned`: 2 instance(s), lines 1242, 1286
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 728
- `print_banned_production_logging`: 75 instance(s), lines 285, 325, 410, 411, 412, 413, 431, 473, 484, 660, 686, 729, 734, 739, 915, 916, 917, 918, 919, 920, 921, 930, 937, 940, 942, ... (+50 more)

### `scripts/gpu_dock_batched.py`
- `bare_except_banned`: 1 instance(s), lines 958
- `json_dump_banned_for_analytical_data`: 2 instance(s), lines 352, 579
- `print_banned_production_logging`: 55 instance(s), lines 227, 312, 313, 314, 315, 333, 462, 511, 537, 580, 585, 590, 766, 767, 768, 769, 770, 771, 772, 781, 788, 791, 793, 796, 921, ... (+30 more)

### `scripts/gpu_dock_fast.py`
- `print_banned_production_logging`: 30 instance(s), lines 115, 116, 117, 118, 119, 120, 121, 122, 125, 127, 130, 137, 140, 149, 150, 151, 165, 166, 167, 169, 170, 171, 172, 95, 98, ... (+5 more)

### `scripts/growth_vector_map.py`
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 360
- `print_banned_production_logging`: 2 instance(s), lines 361, 364

### `scripts/interchain_contacts.py`
- `print_banned_production_logging`: 19 instance(s), lines 350, 351, 352, 353, 378, 379, 380, 381, 382, 383, 392, 394, 395, 396, 425, 445, 355, 386, 388

### `scripts/interfaces/anchor_point.py`
- `banned_pickle_serialization`: 1 instance(s), lines 14

### `scripts/interfaces/consensus_site.py`
- `banned_pickle_serialization`: 1 instance(s), lines 16

### `scripts/interfaces/contact_reorg_result.py`
- `banned_pickle_serialization`: 1 instance(s), lines 13

### `scripts/interfaces/design_brief.py`
- `banned_pickle_serialization`: 1 instance(s), lines 13

### `scripts/interfaces/docking_result.py`
- `banned_pickle_serialization`: 1 instance(s), lines 11

### `scripts/interfaces/ensemble_score.py`
- `banned_pickle_serialization`: 1 instance(s), lines 14

### `scripts/interfaces/explicit_solvent_result.py`
- `banned_pickle_serialization`: 1 instance(s), lines 13

### `scripts/interfaces/fep_result.py`
- `banned_pickle_serialization`: 1 instance(s), lines 9

### `scripts/interfaces/filtered_candidate.py`
- `banned_pickle_serialization`: 1 instance(s), lines 11

### `scripts/interfaces/gating_result.py`
- `banned_pickle_serialization`: 1 instance(s), lines 14

### `scripts/interfaces/generated_molecule.py`
- `banned_pickle_serialization`: 1 instance(s), lines 9

### `scripts/interfaces/growth_vector.py`
- `banned_pickle_serialization`: 1 instance(s), lines 16

### `scripts/interfaces/membrane_system.py`
- `banned_pickle_serialization`: 1 instance(s), lines 13

### `scripts/interfaces/pipeline_config.py`
- `banned_pickle_serialization`: 1 instance(s), lines 10
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 152

### `scripts/interfaces/pocket_dynamics.py`
- `banned_pickle_serialization`: 1 instance(s), lines 13

### `scripts/interfaces/pocket_profile.py`
- `banned_pickle_serialization`: 1 instance(s), lines 13

### `scripts/interfaces/residue_mapping.py`
- `banned_pickle_serialization`: 1 instance(s), lines 14

### `scripts/interfaces/response_profile.py`
- `banned_pickle_serialization`: 1 instance(s), lines 14

### `scripts/interfaces/site_ranking.py`
- `banned_pickle_serialization`: 1 instance(s), lines 14

### `scripts/interfaces/site_spike_view.py`
- `broad_exception_banned`: 9 instance(s), lines 189, 194, 217, 224, 235, 289, 721, 843, 229
- `direct_parquet_writer_banned`: 1 instance(s), lines 302

### `scripts/interfaces/spike_pharmacophore.py`
- `banned_pickle_serialization`: 1 instance(s), lines 20

### `scripts/interfaces/tautomer_state.py`
- `banned_pickle_serialization`: 1 instance(s), lines 15

### `scripts/interfaces/viewer_payload.py`
- `banned_pickle_serialization`: 1 instance(s), lines 13

### `scripts/interfaces/water_map.py`
- `banned_pickle_serialization`: 1 instance(s), lines 14

### `scripts/json_to_pdb.py`
- `print_banned_production_logging`: 11 instance(s), lines 15, 17, 81, 97, 111, 112, 113, 114, 115, 86, 87

### `scripts/kcc_validation_v2.py`
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 350
- `print_banned_production_logging`: 4 instance(s), lines 351, 365, 283, 356

### `scripts/managed-agents/r2_upload_webhook.py`
- `broad_exception_banned`: 1 instance(s), lines 123
- `print_banned_production_logging`: 12 instance(s), lines 104, 203, 204, 205, 206, 32, 75, 83, 122, 191, 124, 212

### `scripts/managed-agents/setup_agents.py`
- `broad_exception_banned`: 1 instance(s), lines 443
- `print_banned_production_logging`: 29 instance(s), lines 298, 317, 318, 319, 326, 431, 432, 433, 434, 47, 270, 273, 275, 279, 283, 285, 289, 293, 295, 314, 428, 440, 464, 465, 442, ... (+4 more)

### `scripts/measure_noise_floor.py`
- `broad_exception_banned`: 8 instance(s), lines 123, 185, 218, 234, 272, 270, 325, 158
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 76
- `print_banned_production_logging`: 6 instance(s), lines 563, 564, 565, 566, 567, 568

### `scripts/mmgbsa_rescore.py`
- `broad_exception_banned`: 2 instance(s), lines 142, 217
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 224
- `print_banned_production_logging`: 15 instance(s), lines 195, 196, 197, 198, 199, 254, 255, 256, 204, 209, 212, 213, 214, 215, 218

### `scripts/multichain_preprocessor.py`
- `broad_exception_banned`: 2 instance(s), lines 203, 306
- `print_banned_production_logging`: 51 instance(s), lines 184, 288, 339, 340, 341, 342, 353, 375, 379, 383, 403, 435, 436, 437, 438, 439, 440, 456, 457, 461, 465, 563, 567, 124, 125, ... (+26 more)

### `scripts/numbering_audit.py`
- `banned_text_table_io`: 1 instance(s), lines 25
- `print_banned_production_logging`: 9 instance(s), lines 175, 176, 177, 178, 181, 73, 74, 79, 180

### `scripts/p2rank_rescore.py`
- `banned_text_table_io`: 1 instance(s), lines 20
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 176
- `print_banned_production_logging`: 7 instance(s), lines 111, 113, 172, 180, 181, 187, 186

### `scripts/pharmacophore_extract.py`
- `print_banned_production_logging`: 34 instance(s), lines 96, 154, 160, 161, 167, 168, 171, 176, 179, 199, 201, 211, 262, 265, 266, 267, 268, 269, 270, 271, 272, 276, 277, 278, 279, ... (+9 more)

### `scripts/phase_manifold_ranker.py`
- `banned_dataframe_engine`: 1 instance(s), lines 10
- `broad_exception_banned`: 3 instance(s), lines 70, 79, 92
- `print_banned_production_logging`: 14 instance(s), lines 700, 701, 702, 703, 704, 705, 706, 857, 858, 859, 860, 861, 862, 864

### `scripts/pocket_fusion.py`
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 398
- `print_banned_production_logging`: 13 instance(s), lines 336, 339, 351, 355, 400, 380, 381, 382, 383, 332, 386, 388, 390

### `scripts/pocket_profile_builder.py`
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 212
- `print_banned_production_logging`: 2 instance(s), lines 213, 216

### `scripts/post_dock_analysis.py`
- `broad_exception_banned`: 1 instance(s), lines 836
- `print_banned_production_logging`: 36 instance(s), lines 770, 771, 772, 775, 779, 780, 795, 798, 850, 860, 878, 886, 895, 901, 904, 905, 906, 914, 915, 916, 917, 623, 626, 792, 805, ... (+11 more)

### `scripts/postprocess_twin.py`
- `broad_exception_banned`: 3 instance(s), lines 129, 1126, 1783
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 131
- `print_banned_production_logging`: 1 instance(s), lines 107

### `scripts/prepare_production_topologies.py`
- `bare_except_banned`: 1 instance(s), lines 74
- `broad_exception_banned`: 1 instance(s), lines 128
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 231
- `print_banned_production_logging`: 45 instance(s), lines 135, 136, 137, 138, 156, 157, 158, 159, 160, 180, 181, 182, 183, 191, 192, 193, 194, 195, 209, 210, 211, 212, 213, 214, 215, ... (+20 more)

### `scripts/prepare_protein.py`
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 264
- `print_banned_production_logging`: 27 instance(s), lines 29, 79, 100, 103, 111, 119, 218, 222, 262, 266, 16, 25, 26, 36, 50, 68, 70, 74, 332, 42, 152, 236, 238, 166, 181, ... (+2 more)

### `scripts/prepare_protien.py`
- `python_syntax_error`: 1 instance(s), lines 2

### `scripts/prepare_structure_openmm.py`
- `print_banned_production_logging`: 43 instance(s), lines 55, 56, 57, 58, 59, 60, 61, 64, 81, 82, 86, 104, 116, 119, 142, 150, 151, 156, 166, 169, 170, 171, 172, 173, 174, ... (+18 more)

### `scripts/preprocessing/membrane_builder.py`
- `broad_exception_banned`: 1 instance(s), lines 74
- `print_banned_production_logging`: 3 instance(s), lines 340, 349, 351

### `scripts/preprocessing/protein_fixer.py`
- `print_banned_production_logging`: 1 instance(s), lines 331

### `scripts/preprocessing/target_classifier.py`
- `broad_exception_banned`: 3 instance(s), lines 89, 125, 160
- `print_banned_production_logging`: 1 instance(s), lines 344

### `scripts/preprocessing/tautomer_enumeration.py`
- `broad_exception_banned`: 2 instance(s), lines 77, 150
- `print_banned_production_logging`: 1 instance(s), lines 329

### `scripts/prism-clean.py`
- `print_banned_production_logging`: 10 instance(s), lines 64, 65, 66, 67, 68, 76, 17, 51, 71, 82

### `scripts/prism-corpus-status.py`
- `print_banned_production_logging`: 1 instance(s), lines 711

### `scripts/prism-ground-truth.py`
- `print_banned_production_logging`: 11 instance(s), lines 461, 462, 479, 448, 466, 467, 471, 473, 474, 476, 478

### `scripts/prism-lookup-residue.py`
- `print_banned_production_logging`: 7 instance(s), lines 18, 43, 45, 55, 72, 76, 67

### `scripts/prism-merge-chains.py`
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 142
- `print_banned_production_logging`: 7 instance(s), lines 145, 151, 152, 23, 39, 127, 147

### `scripts/prism-postflight.py`
- `print_banned_production_logging`: 37 instance(s), lines 179, 180, 181, 182, 183, 216, 271, 272, 294, 300, 334, 335, 336, 340, 365, 185, 189, 205, 208, 213, 276, 291, 306, 326, 387, ... (+12 more)

### `scripts/prism-preflight.py`
- `print_banned_production_logging`: 19 instance(s), lines 137, 138, 139, 140, 141, 142, 143, 156, 43, 145, 148, 153, 163, 169, 171, 173, 51, 61, 152

### `scripts/prism-r2-sync/prism_spike_watcher.py`
- `broad_exception_banned`: 6 instance(s), lines 254, 281, 292, 540, 556, 828
- `direct_parquet_writer_banned`: 4 instance(s), lines 397, 374, 507, 472

### `scripts/prism_canonical.py`
- `json_dump_banned_for_analytical_data`: 2 instance(s), lines 296, 298
- `print_banned_production_logging`: 30 instance(s), lines 182, 186, 191, 200, 204, 209, 215, 230, 240, 248, 268, 280, 300, 305, 306, 309, 310, 311, 312, 313, 314, 315, 233, 243, 253, ... (+5 more)

### `scripts/prism_chronic_durability_bridge.py`
- `direct_parquet_writer_banned`: 2 instance(s), lines 249, 290
- `print_banned_production_logging`: 2 instance(s), lines 352, 353

### `scripts/prism_kcc_decoder.py`
- `direct_parquet_writer_banned`: 3 instance(s), lines 281, 282, 283
- `print_banned_production_logging`: 2 instance(s), lines 320, 321

### `scripts/prism_pipeline.py`
- `banned_determinant`: 1 instance(s), lines 242
- `broad_exception_banned`: 1 instance(s), lines 630
- `json_dump_banned_for_analytical_data`: 2 instance(s), lines 351, 558
- `print_banned_production_logging`: 51 instance(s), lines 103, 104, 105, 122, 127, 166, 167, 168, 169, 170, 196, 197, 201, 267, 268, 276, 277, 290, 308, 309, 321, 322, 329, 354, 433, ... (+26 more)

### `scripts/prism_pipeline_batch.py`
- `broad_exception_banned`: 3 instance(s), lines 140, 519, 360
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 433
- `print_banned_production_logging`: 47 instance(s), lines 199, 200, 218, 221, 285, 299, 324, 325, 326, 327, 328, 329, 330, 331, 332, 336, 337, 338, 368, 371, 376, 377, 378, 379, 397, ... (+22 more)

### `scripts/prism_ranker.py`
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 74
- `print_banned_production_logging`: 1 instance(s), lines 77

### `scripts/prism_replicate.py`
- `print_banned_production_logging`: 22 instance(s), lines 114, 141, 142, 143, 144, 193, 39, 77, 94, 111, 152, 173, 189, 200, 201, 203, 159, 165, 170, 178, 180, 163

### `scripts/prism_spike_event_integrator.py`
- `direct_parquet_writer_banned`: 5 instance(s), lines 579, 747, 754, 657, 685
- `print_banned_production_logging`: 3 instance(s), lines 853, 855, 859

### `scripts/prism_spike_watcher.py`
- `broad_exception_banned`: 7 instance(s), lines 262, 289, 300, 425, 441, 493, 870
- `direct_parquet_writer_banned`: 2 instance(s), lines 392, 357

### `scripts/prism_twin_forensic_schema_audit.py`
- `broad_exception_banned`: 8 instance(s), lines 22, 199, 224, 276, 310, 324, 353, 475
- `print_banned_production_logging`: 2 instance(s), lines 1158, 1159

### `scripts/production/apply_phase4_migration.py`
- `print_banned_production_logging`: 15 instance(s), lines 114, 115, 116, 143, 153, 161, 163, 110, 141, 145, 119, 147, 148, 157, 159

### `scripts/production/curate_prism5000_chain_targets.py`
- `banned_text_table_io`: 1 instance(s), lines 20
- `print_banned_production_logging`: 14 instance(s), lines 786, 787, 794, 814, 839, 848, 859, 842, 513, 809, 286, 537, 833, 855

### `scripts/production/export_v4_snapshot.py`
- `banned_dataframe_engine`: 1 instance(s), lines 103
- `broad_exception_banned`: 4 instance(s), lines 80, 144, 211, 177
- `direct_parquet_writer_banned`: 7 instance(s), lines 188, 189, 190, 191, 192, 193, 186
- `print_banned_production_logging`: 10 instance(s), lines 36, 113, 332, 333, 334, 335, 119, 122, 130, 212

### `scripts/production/finalize_prism5000_ready_set.py`
- `banned_text_table_io`: 1 instance(s), lines 17
- `print_banned_production_logging`: 1 instance(s), lines 135

### `scripts/production/prepare_prism5000_chain_targets.py`
- `broad_exception_banned`: 3 instance(s), lines 181, 351, 288
- `print_banned_production_logging`: 3 instance(s), lines 393, 448, 423

### `scripts/production/prism-strip-hect.py`
- `print_banned_production_logging`: 15 instance(s), lines 324, 325, 326, 327, 328, 329, 330, 331, 332, 248, 291, 293, 275, 256, 261

### `scripts/production/stale_row_purge.py`
- `broad_exception_banned`: 3 instance(s), lines 168, 267, 71
- `print_banned_production_logging`: 10 instance(s), lines 303, 296, 299, 310, 315, 319, 325, 333, 172, 321

### `scripts/production/validate_v4_contract.py`
- `broad_exception_banned`: 3 instance(s), lines 131, 145, 326
- `print_banned_production_logging`: 12 instance(s), lines 30, 352, 353, 355, 356, 357, 358, 359, 360, 370, 374, 372

### `scripts/production/w2_dcc_pin.py`
- `broad_exception_banned`: 2 instance(s), lines 274, 58
- `print_banned_production_logging`: 8 instance(s), lines 379, 374, 376, 384, 390, 396, 257, 283

### `scripts/production/w3b_event_aggregates_pin.py`
- `broad_exception_banned`: 5 instance(s), lines 139, 174, 282, 296, 60
- `print_banned_production_logging`: 8 instance(s), lines 420, 415, 417, 425, 431, 441, 340, 378

### `scripts/production/w4_pin_runtime.py`
- `print_banned_production_logging`: 13 instance(s), lines 135, 136, 137, 138, 139, 141, 143, 105, 108, 111, 128, 133, 146

### `scripts/prt70_panel_sweep.py`
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 663
- `print_banned_production_logging`: 9 instance(s), lines 570, 571, 572, 665, 681, 691, 575, 668, 684

### `scripts/quarantine/aggregate_site_tags.py`
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 1284
- `print_banned_production_logging`: 1 instance(s), lines 1279

### `scripts/quarantine/align_6p8y_to_6oim.py`
- `print_banned_production_logging`: 5 instance(s), lines 68, 69, 70, 75, 52

### `scripts/quarantine/arrow_to_legacy_json.py`
- `print_banned_production_logging`: 1 instance(s), lines 192

### `scripts/quarantine/audit_compute_kras_holo_pockets.py`
- `print_banned_production_logging`: 4 instance(s), lines 152, 154, 110, 114

### `scripts/quarantine/audit_json_parquet_equivalence.py`
- `broad_exception_banned`: 3 instance(s), lines 174, 297, 346
- `print_banned_production_logging`: 46 instance(s), lines 370, 371, 372, 373, 374, 375, 376, 379, 382, 385, 393, 394, 416, 452, 453, 454, 458, 533, 534, 538, 539, 540, 378, 381, 384, ... (+21 more)

### `scripts/quarantine/audit_manifest_gt.py`
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 281
- `print_banned_production_logging`: 18 instance(s), lines 220, 224, 227, 242, 245, 265, 266, 267, 268, 269, 270, 271, 272, 288, 296, 304, 276, 238

### `scripts/quarantine/augment_therm_ccns_join.py`
- `banned_text_table_io`: 1 instance(s), lines 10
- `print_banned_production_logging`: 4 instance(s), lines 185, 142, 152, 183

### `scripts/quarantine/b_no_therm_feature_gap.py`
- `print_banned_production_logging`: 32 instance(s), lines 213, 214, 215, 242, 243, 244, 245, 250, 257, 272, 273, 274, 276, 277, 278, 279, 281, 282, 296, 299, 219, 220, 223, 225, 226, ... (+7 more)

### `scripts/quarantine/build_m1_panel.py`
- `print_banned_production_logging`: 8 instance(s), lines 108, 109, 113, 114, 115, 116, 163, 111

### `scripts/quarantine/build_never_bound_manifest.py`
- `json_dump_banned_for_analytical_data`: 4 instance(s), lines 421, 470, 453, 463
- `print_banned_production_logging`: 34 instance(s), lines 117, 146, 152, 189, 291, 292, 319, 320, 321, 322, 353, 387, 428, 429, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, ... (+9 more)

### `scripts/quarantine/build_true_apo_manifest.py`
- `banned_pickle_serialization`: 1 instance(s), lines 26
- `json_dump_banned_for_analytical_data`: 4 instance(s), lines 678, 161, 606, 241
- `print_banned_production_logging`: 73 instance(s), lines 69, 70, 98, 265, 395, 396, 397, 398, 426, 427, 440, 462, 469, 525, 541, 577, 608, 679, 686, 696, 697, 698, 699, 700, 701, ... (+48 more)

### `scripts/quarantine/canonical_dcc_audit.py`
- `broad_exception_banned`: 1 instance(s), lines 121
- `print_banned_production_logging`: 22 instance(s), lines 216, 217, 240, 241, 242, 243, 244, 245, 246, 273, 274, 227, 249, 250, 255, 256, 262, 263, 252, 258, 265, 266

### `scripts/quarantine/classifier_audit.py`
- `print_banned_production_logging`: 38 instance(s), lines 302, 303, 304, 306, 307, 320, 322, 323, 324, 327, 329, 330, 331, 332, 335, 345, 347, 348, 349, 350, 351, 353, 365, 366, 367, ... (+13 more)

### `scripts/quarantine/classifier_audit_expanded.py`
- `banned_text_table_io`: 1 instance(s), lines 22
- `broad_exception_banned`: 1 instance(s), lines 42
- `print_banned_production_logging`: 16 instance(s), lines 233, 234, 235, 236, 238, 249, 250, 251, 252, 253, 254, 257, 260, 261, 262, 240

### `scripts/quarantine/completeness_certifier.py`
- `banned_text_table_io`: 1 instance(s), lines 22
- `broad_exception_banned`: 8 instance(s), lines 173, 202, 319, 354, 368, 382, 399, 374
- `print_banned_production_logging`: 6 instance(s), lines 700, 701, 702, 677, 681, 704

### `scripts/quarantine/d3_parity_harness.py`
- `print_banned_production_logging`: 1 instance(s), lines 211

### `scripts/quarantine/d3_timing_harness.py`
- `print_banned_production_logging`: 4 instance(s), lines 132, 133, 134, 147

### `scripts/quarantine/d4_rank1_harness.py`
- `print_banned_production_logging`: 6 instance(s), lines 248, 249, 250, 251, 252, 261

### `scripts/quarantine/d5_step_e_harness.py`
- `print_banned_production_logging`: 11 instance(s), lines 193, 194, 195, 196, 197, 167, 168, 179, 158, 159, 170

### `scripts/quarantine/d5_v2_phase2_harness.py`
- `print_banned_production_logging`: 5 instance(s), lines 135, 136, 137, 138, 139

### `scripts/quarantine/d5_wave1_extract_all_features_harness.py`
- `broad_exception_banned`: 1 instance(s), lines 171
- `print_banned_production_logging`: 12 instance(s), lines 217, 218, 219, 220, 221, 222, 223, 202, 174, 187, 173, 186

### `scripts/quarantine/d5_wave1_train_v006_harness.py`
- `broad_exception_banned`: 2 instance(s), lines 169, 176
- `print_banned_production_logging`: 13 instance(s), lines 205, 206, 207, 208, 209, 210, 211, 178, 179, 193, 171, 177, 196

### `scripts/quarantine/docx_one_panel_per_page.py`
- `broad_exception_banned`: 1 instance(s), lines 100
- `print_banned_production_logging`: 9 instance(s), lines 103, 104, 167, 163, 115, 130, 133, 101, 158

### `scripts/quarantine/docx_split_panels_per_page.py`
- `bare_except_banned`: 1 instance(s), lines 200
- `print_banned_production_logging`: 4 instance(s), lines 128, 201, 137, 194

### `scripts/quarantine/engine_full_harvest.py`
- `banned_text_table_io`: 1 instance(s), lines 18
- `broad_exception_banned`: 6 instance(s), lines 168, 192, 280, 314, 287, 467
- `print_banned_production_logging`: 7 instance(s), lines 723, 724, 725, 726, 727, 728, 688

### `scripts/quarantine/enrich_spike_arrow.py`
- `direct_parquet_writer_banned`: 1 instance(s), lines 371
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 327
- `print_banned_production_logging`: 15 instance(s), lines 64, 68, 87, 229, 236, 268, 328, 345, 346, 347, 348, 400, 342, 369, 372

### `scripts/quarantine/enrich_tem1_report.py`
- `json_dump_banned_for_analytical_data`: 4 instance(s), lines 174, 309, 447, 802
- `print_banned_production_logging`: 61 instance(s), lines 100, 134, 176, 201, 255, 263, 290, 311, 334, 449, 457, 519, 532, 595, 603, 677, 686, 805, 806, 807, 808, 809, 810, 811, 812, ... (+36 more)

### `scripts/quarantine/fill_prism_manifold_residues.py`
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 199
- `print_banned_production_logging`: 12 instance(s), lines 90, 91, 164, 165, 200, 94, 112, 156, 157, 158, 127, 190

### `scripts/quarantine/fix_tem1_v3.py`
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 694
- `print_banned_production_logging`: 68 instance(s), lines 39, 57, 58, 63, 196, 197, 202, 205, 224, 230, 257, 258, 308, 327, 328, 379, 380, 385, 439, 558, 561, 566, 591, 603, 663, ... (+43 more)

### `scripts/quarantine/freeze_package_manifest.py`
- `print_banned_production_logging`: 7 instance(s), lines 116, 120, 121, 122, 123, 124, 119

### `scripts/quarantine/full_schema_lossless_validator.py`
- `banned_text_table_io`: 1 instance(s), lines 35
- `broad_exception_banned`: 2 instance(s), lines 207, 249
- `print_banned_production_logging`: 25 instance(s), lines 458, 459, 460, 461, 464, 465, 468, 469, 470, 471, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 463, 467, 473, 221, 475

### `scripts/quarantine/gate_a_validator.py`
- `banned_text_table_io`: 1 instance(s), lines 18
- `print_banned_production_logging`: 15 instance(s), lines 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 351

### `scripts/quarantine/generate_tem1_report.py`
- `broad_exception_banned`: 7 instance(s), lines 240, 238, 386, 474, 485, 352, 1073
- `json_dump_banned_for_analytical_data`: 4 instance(s), lines 1060, 1111, 1148, 1173
- `print_banned_production_logging`: 98 instance(s), lines 95, 96, 97, 102, 140, 141, 142, 143, 164, 177, 248, 367, 371, 390, 391, 401, 453, 455, 460, 637, 888, 893, 936, 969, 1061, ... (+73 more)

### `scripts/quarantine/ghost_v2_native_spatial_probe.py`
- `broad_exception_banned`: 1 instance(s), lines 122
- `print_banned_production_logging`: 2 instance(s), lines 215, 216

### `scripts/quarantine/incremental_delete.py`
- `broad_exception_banned`: 4 instance(s), lines 44, 55, 28, 97
- `print_banned_production_logging`: 1 instance(s), lines 111

### `scripts/quarantine/insert_pfr_section_v6.py`
- `print_banned_production_logging`: 6 instance(s), lines 195, 273, 301, 296, 298, 180

### `scripts/quarantine/investigate_json_only_sites.py`
- `broad_exception_banned`: 3 instance(s), lines 138, 118, 234
- `print_banned_production_logging`: 22 instance(s), lines 168, 169, 170, 180, 185, 186, 187, 189, 241, 242, 247, 264, 172, 183, 249, 254, 259, 256, 261, 119, 232, 235

### `scripts/quarantine/m1_1_manifest.py`
- `broad_exception_banned`: 1 instance(s), lines 57
- `print_banned_production_logging`: 8 instance(s), lines 102, 104, 111, 112, 113, 114, 121, 106

### `scripts/quarantine/m1_2_25_audit_driver.py`
- `broad_exception_banned`: 1 instance(s), lines 33
- `print_banned_production_logging`: 15 instance(s), lines 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 596, 597, 577

### `scripts/quarantine/m1_2_decision_notes.py`
- `banned_text_table_io`: 1 instance(s), lines 15
- `print_banned_production_logging`: 6 instance(s), lines 364, 365, 366, 367, 368, 330

### `scripts/quarantine/m1_2b_note_hygiene.py`
- `banned_text_table_io`: 1 instance(s), lines 24
- `broad_exception_banned`: 2 instance(s), lines 201, 176
- `print_banned_production_logging`: 9 instance(s), lines 557, 558, 559, 560, 548, 562, 566, 564, 568

### `scripts/quarantine/m1_2c_cleanup.py`
- `banned_text_table_io`: 1 instance(s), lines 13
- `broad_exception_banned`: 2 instance(s), lines 40, 62
- `print_banned_production_logging`: 12 instance(s), lines 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359

### `scripts/quarantine/m1_ablation.py`
- `broad_exception_banned`: 1 instance(s), lines 376
- `print_banned_production_logging`: 25 instance(s), lines 453, 454, 455, 468, 469, 470, 471, 472, 473, 478, 479, 480, 482, 483, 484, 485, 490, 498, 457, 459, 476, 487, 504, 505, 462

### `scripts/quarantine/m1_v4_postflight_check.py`
- `print_banned_production_logging`: 8 instance(s), lines 276, 297, 322, 325, 326, 282, 285, 308

### `scripts/quarantine/per_target_completion_block.py`
- `broad_exception_banned`: 1 instance(s), lines 50
- `print_banned_production_logging`: 1 instance(s), lines 171

### `scripts/quarantine/pfr_build_10A_10C.py`
- `banned_text_table_io`: 1 instance(s), lines 8
- `print_banned_production_logging`: 2 instance(s), lines 130, 209

### `scripts/quarantine/pfr_build_feature_csvs.py`
- `banned_text_table_io`: 1 instance(s), lines 25
- `broad_exception_banned`: 4 instance(s), lines 192, 196, 210, 147
- `print_banned_production_logging`: 8 instance(s), lines 343, 406, 470, 311, 317, 322, 327, 211

### `scripts/quarantine/pfr_composite_10B.py`
- `print_banned_production_logging`: 1 instance(s), lines 73

### `scripts/quarantine/pfr_render_panels.py`
- `banned_text_table_io`: 1 instance(s), lines 13
- `broad_exception_banned`: 2 instance(s), lines 71, 203

### `scripts/quarantine/prism-aggregate-sites.py`
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 1055

### `scripts/quarantine/prism_engine_prov.py`
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 518
- `print_banned_production_logging`: 11 instance(s), lines 527, 531, 532, 537, 540, 541, 542, 544, 530, 536, 539

### `scripts/quarantine/prism_manifold_shell_validator.py`
- `banned_determinant`: 1 instance(s), lines 150
- `banned_text_table_io`: 1 instance(s), lines 30
- `broad_exception_banned`: 2 instance(s), lines 344, 442
- `print_banned_production_logging`: 41 instance(s), lines 958, 959, 960, 961, 964, 968, 971, 1002, 1113, 1115, 1119, 1121, 1150, 1153, 1161, 1166, 1186, 1187, 1188, 1189, 1190, 1191, 1192, 1193, 1194, ... (+16 more)

### `scripts/quarantine/prism_postrun_seqalign.py`
- `json_dump_banned_for_analytical_data`: 7 instance(s), lines 151, 171, 192, 217, 237, 257, 277
- `print_banned_production_logging`: 27 instance(s), lines 368, 378, 381, 427, 507, 509, 510, 511, 371, 372, 393, 406, 419, 433, 443, 453, 496, 437, 447, 457, 464, 475, 486, 500, 468, ... (+2 more)

### `scripts/quarantine/prism_prov.py`
- `json_dump_banned_for_analytical_data`: 2 instance(s), lines 456, 212
- `print_banned_production_logging`: 3 instance(s), lines 473, 474, 475

### `scripts/quarantine/prism_pub_baseline_validator.py`
- `banned_determinant`: 1 instance(s), lines 181
- `banned_text_table_io`: 1 instance(s), lines 32
- `broad_exception_banned`: 2 instance(s), lines 381, 480
- `print_banned_production_logging`: 47 instance(s), lines 1291, 1292, 1293, 1294, 1295, 1298, 1302, 1305, 1346, 1347, 1349, 1350, 1352, 1355, 1466, 1480, 1484, 1487, 1489, 1493, 1518, 1519, 1520, 1521, 1522, ... (+22 more)

### `scripts/quarantine/process_cryptobank.py`
- `banned_dataframe_engine`: 1 instance(s), lines 31
- `banned_pickle_serialization`: 1 instance(s), lines 20
- `json_dump_banned_for_analytical_data`: 2 instance(s), lines 402, 370
- `print_banned_production_logging`: 32 instance(s), lines 147, 149, 186, 224, 246, 251, 374, 410, 411, 412, 413, 414, 415, 427, 428, 429, 430, 189, 232, 238, 283, 305, 311, 266, 274, ... (+7 more)

### `scripts/quarantine/r2_manifest_enrich_and_proof.py`
- `broad_exception_banned`: 1 instance(s), lines 57
- `print_banned_production_logging`: 19 instance(s), lines 132, 133, 134, 135, 136, 137, 138, 141, 142, 148, 149, 154, 155, 140, 145, 151, 153, 157, 160

### `scripts/quarantine/r2_offload_completed_targets.py`
- `banned_text_table_io`: 1 instance(s), lines 22
- `broad_exception_banned`: 6 instance(s), lines 237, 251, 267, 412, 308, 400
- `print_banned_production_logging`: 18 instance(s), lines 337, 405, 344, 376, 431, 434, 411, 433, 439, 440, 444, 445, 446, 450, 451, 448, 453, 455

### `scripts/quarantine/rebuild_patent_docx.py`
- `print_banned_production_logging`: 2 instance(s), lines 211, 212

### `scripts/quarantine/rebuild_patent_docx_v2.py`
- `print_banned_production_logging`: 3 instance(s), lines 397, 398, 402

### `scripts/quarantine/remap_topo_resids.py`
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 213
- `print_banned_production_logging`: 23 instance(s), lines 116, 119, 148, 152, 153, 154, 157, 161, 188, 189, 190, 191, 193, 215, 216, 217, 218, 132, 143, 197, 209, 140, 186

### `scripts/quarantine/reranker_v4_six_pillar_manifold.py`
- `print_banned_production_logging`: 13 instance(s), lines 182, 183, 185, 213, 216, 148, 163, 174, 219, 226, 227, 239, 233

### `scripts/quarantine/run_metadata_writer.py`
- `print_banned_production_logging`: 1 instance(s), lines 155

### `scripts/quarantine/run_stages.py`
- `banned_determinant`: 1 instance(s), lines 419
- `broad_exception_banned`: 9 instance(s), lines 74, 148, 223, 522, 702, 870, 1450, 896, 1171
- `json_dump_banned_for_analytical_data`: 7 instance(s), lines 574, 1096, 1373, 1185, 1202, 1226, 1246
- `print_banned_production_logging`: 7 instance(s), lines 1463, 704, 1440, 75, 1458, 1460, 1452

### `scripts/quarantine/site_vs_holo_strict.py`
- `print_banned_production_logging`: 23 instance(s), lines 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 309, 311

### `scripts/quarantine/spike_metadata_inventory.py`
- `broad_exception_banned`: 2 instance(s), lines 58, 82
- `print_banned_production_logging`: 34 instance(s), lines 492, 493, 494, 497, 498, 499, 500, 519, 520, 521, 522, 523, 524, 525, 536, 537, 538, 552, 553, 554, 555, 556, 564, 565, 496, ... (+9 more)

### `scripts/quarantine/twin10_audit.py`
- `broad_exception_banned`: 3 instance(s), lines 249, 329, 504
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 566
- `print_banned_production_logging`: 8 instance(s), lines 524, 525, 552, 567, 545, 550, 554, 549

### `scripts/quarantine/validate_dcc_smoke5.py`
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 269
- `print_banned_production_logging`: 33 instance(s), lines 150, 170, 171, 178, 188, 189, 190, 221, 222, 223, 224, 225, 235, 236, 237, 242, 253, 254, 255, 256, 257, 258, 261, 276, 153, ... (+8 more)

### `scripts/quarantine/validate_v2_7c8r.py`
- `print_banned_production_logging`: 8 instance(s), lines 43, 44, 45, 46, 52, 53, 29, 62

### `scripts/quarantine/verify_twin_provenance.py`
- `broad_exception_banned`: 4 instance(s), lines 51, 71, 89, 142
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 258
- `print_banned_production_logging`: 26 instance(s), lines 190, 191, 192, 195, 211, 212, 218, 221, 227, 228, 239, 242, 243, 244, 245, 246, 247, 248, 252, 253, 184, 202, 274, 224, 208, ... (+1 more)

### `scripts/quarantine/w3b_shard_targets.py`
- `print_banned_production_logging`: 4 instance(s), lines 49, 32, 51, 53

### `scripts/quarantine/wrn_1522_spatial_validation.py`
- `print_banned_production_logging`: 23 instance(s), lines 221, 222, 223, 224, 225, 226, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 243, 244, 245, 246, 229, 242

### `scripts/quarantine/wrn_1522_strict_verify.py`
- `print_banned_production_logging`: 49 instance(s), lines 85, 135, 136, 137, 140, 144, 145, 146, 151, 152, 153, 154, 163, 164, 165, 166, 167, 185, 186, 187, 188, 204, 205, 206, 207, ... (+24 more)

### `scripts/rank_search.py`
- `print_banned_production_logging`: 19 instance(s), lines 140, 164, 165, 166, 176, 177, 180, 181, 193, 194, 143, 179, 184, 189, 197, 91, 98, 173, 188

### `scripts/ranking_lab.py`
- `banned_text_table_io`: 1 instance(s), lines 18
- `print_banned_production_logging`: 10 instance(s), lines 196, 202, 236, 240, 241, 258, 239, 244, 250, 249

### `scripts/ranking_lab_v2.py`
- `banned_text_table_io`: 1 instance(s), lines 17
- `print_banned_production_logging`: 9 instance(s), lines 190, 224, 228, 229, 243, 227, 232, 236, 235

### `scripts/rerank_sites.py`
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 121
- `print_banned_production_logging`: 11 instance(s), lines 173, 174, 175, 176, 194, 195, 197, 170, 188, 166, 182

### `scripts/response_selectivity.py`
- `broad_exception_banned`: 2 instance(s), lines 90, 528
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 586
- `print_banned_production_logging`: 3 instance(s), lines 587, 591, 602

### `scripts/run_baselines.py`
- `banned_text_table_io`: 1 instance(s), lines 39
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 413
- `print_banned_production_logging`: 22 instance(s), lines 220, 221, 222, 223, 224, 260, 261, 270, 271, 305, 332, 377, 380, 389, 392, 401, 414, 258, 294, 302, 313, 320

### `scripts/run_bench10.py`
- `banned_text_table_io`: 1 instance(s), lines 22
- `broad_exception_banned`: 2 instance(s), lines 341, 460
- `json_dump_banned_for_analytical_data`: 5 instance(s), lines 273, 334, 353, 504, 396
- `print_banned_production_logging`: 1 instance(s), lines 90

### `scripts/run_cryptobench.py`
- `bare_except_banned`: 1 instance(s), lines 230
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 258
- `print_banned_production_logging`: 18 instance(s), lines 191, 236, 243, 246, 247, 248, 249, 250, 251, 260, 203, 233, 254, 219, 232, 207, 209, 216

### `scripts/run_hard_targets.py`
- `broad_exception_banned`: 4 instance(s), lines 31, 129, 222, 521
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 417
- `print_banned_production_logging`: 4 instance(s), lines 90, 32, 33, 34

### `scripts/setup_atlas_benchmark.py`
- `banned_text_table_io`: 1 instance(s), lines 11
- `json_dump_banned_for_analytical_data`: 3 instance(s), lines 88, 95, 116
- `print_banned_production_logging`: 28 instance(s), lines 41, 42, 43, 44, 45, 51, 77, 78, 79, 80, 81, 90, 91, 96, 118, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, ... (+3 more)

### `scripts/spike_pharmacophore_map.py`
- `broad_exception_banned`: 2 instance(s), lines 85, 728
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 680
- `print_banned_production_logging`: 30 instance(s), lines 701, 702, 703, 709, 761, 766, 769, 771, 778, 784, 788, 792, 795, 796, 797, 804, 805, 806, 807, 812, 813, 814, 734, 735, 742, ... (+5 more)

### `scripts/stage1_sanitize.py`
- `broad_exception_banned`: 2 instance(s), lines 75, 317
- `print_banned_production_logging`: 26 instance(s), lines 68, 35, 36, 73, 131, 143, 147, 184, 198, 199, 218, 219, 223, 248, 126, 166, 190, 306, 307, 308, 309, 310, 311, 318, 159, ... (+1 more)

### `scripts/stage1_sanitize_amber.py`
- `broad_exception_banned`: 1 instance(s), lines 146
- `print_banned_production_logging`: 20 instance(s), lines 87, 180, 185, 186, 192, 193, 194, 195, 196, 203, 204, 205, 98, 108, 112, 122, 138, 139, 144, 147

### `scripts/stage1_sanitize_hybrid.py`
- `bare_except_banned`: 1 instance(s), lines 170
- `broad_exception_banned`: 2 instance(s), lines 131, 191
- `print_banned_production_logging`: 21 instance(s), lines 223, 224, 276, 282, 283, 284, 285, 286, 291, 292, 92, 127, 146, 183, 254, 133, 156, 189, 193, 247, 241

### `scripts/stage2_topology.py`
- `broad_exception_banned`: 1 instance(s), lines 1116
- `json_dump_banned_for_analytical_data`: 3 instance(s), lines 883, 902, 929
- `print_banned_production_logging`: 48 instance(s), lines 33, 34, 336, 396, 461, 509, 524, 547, 581, 646, 757, 769, 880, 904, 931, 935, 1087, 240, 249, 476, 478, 492, 495, 514, 552, ... (+23 more)

### `scripts/stage2_topology_glycam.py`
- `bare_except_banned`: 1 instance(s), lines 87
- `broad_exception_banned`: 3 instance(s), lines 213, 587, 171
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 461
- `print_banned_production_logging`: 56 instance(s), lines 32, 33, 123, 129, 195, 237, 271, 301, 390, 395, 458, 465, 555, 154, 165, 168, 185, 242, 256, 267, 411, 571, 572, 573, 574, ... (+31 more)

### `scripts/target_config.py`
- `print_banned_production_logging`: 9 instance(s), lines 128, 129, 131, 132, 133, 134, 140, 137, 139

### `scripts/test_coherence_layer.py`
- `print_banned_production_logging`: 7 instance(s), lines 72, 73, 74, 127, 128, 129, 122

### `scripts/test_contact_reorg.py`
- `print_banned_production_logging`: 10 instance(s), lines 168, 169, 184, 186, 204, 205, 219, 181, 212, 217

### `scripts/test_dccm.py`
- `print_banned_production_logging`: 16 instance(s), lines 102, 105, 112, 114, 116, 122, 138, 139, 140, 141, 142, 84, 113, 117, 121, 132

### `scripts/test_four_stage_decision.py`
- `print_banned_production_logging`: 2 instance(s), lines 53, 83

### `scripts/test_probe_panel.py`
- `print_banned_production_logging`: 13 instance(s), lines 155, 156, 157, 203, 208, 209, 210, 211, 212, 185, 200, 178, 188

### `scripts/test_rerank.py`
- `print_banned_production_logging`: 14 instance(s), lines 153, 154, 155, 174, 175, 176, 182, 183, 150, 167, 171, 195, 161, 170

### `scripts/test_target_config.py`
- `print_banned_production_logging`: 1 instance(s), lines 108

### `scripts/tests/test_m1_1_panel_integrity.py`
- `print_banned_production_logging`: 1 instance(s), lines 108

### `scripts/tier2_wasserstein_alignment.py`
- `banned_determinant`: 1 instance(s), lines 84

### `scripts/train_hysteresis_predictor.py`
- `banned_dataframe_engine`: 1 instance(s), lines 29
- `banned_pickle_serialization`: 1 instance(s), lines 22
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 462
- `print_banned_production_logging`: 75 instance(s), lines 226, 227, 228, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 332, 333, 334, 401, 402, 403, 417, 440, 441, 442, 450, ... (+50 more)
- `sklearn_model_import_requires_exemption`: 2 instance(s), lines 31, 31

### `scripts/training/add_temporal_to_npz.py`
- `broad_exception_banned`: 4 instance(s), lines 92, 104, 142, 173
- `print_banned_production_logging`: 3 instance(s), lines 215, 241, 234

### `scripts/training/assemble_v003_bundle.py`
- `broad_exception_banned`: 1 instance(s), lines 231
- `print_banned_production_logging`: 4 instance(s), lines 235, 134, 227, 232

### `scripts/training/cluster_split.py`
- `broad_exception_banned`: 3 instance(s), lines 228, 276, 52
- `print_banned_production_logging`: 9 instance(s), lines 304, 312, 313, 314, 209, 256, 233, 246, 225

### `scripts/training/differential_analysis.py`
- `banned_text_table_io`: 1 instance(s), lines 34
- `broad_exception_banned`: 4 instance(s), lines 104, 134, 155, 224
- `print_banned_production_logging`: 6 instance(s), lines 281, 388, 432, 274, 399, 292

### `scripts/training/extract_all_features.py`
- `broad_exception_banned`: 10 instance(s), lines 115, 189, 285, 384, 429, 588, 628, 673, 734, 829
- `print_banned_production_logging`: 6 instance(s), lines 817, 849, 850, 801, 810, 837

### `scripts/training/extract_channel_features.py`
- `broad_exception_banned`: 4 instance(s), lines 434, 478, 122, 129
- `print_banned_production_logging`: 4 instance(s), lines 513, 551, 553, 540

### `scripts/training/extract_from_r2.py`
- `broad_exception_banned`: 1 instance(s), lines 184
- `print_banned_production_logging`: 5 instance(s), lines 161, 213, 215, 202, 173

### `scripts/training/extract_scan_teacher_artifacts.py`
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 366
- `print_banned_production_logging`: 1 instance(s), lines 367

### `scripts/training/extract_structural_nma_batch.py`
- `broad_exception_banned`: 4 instance(s), lines 41, 47, 58, 64
- `print_banned_production_logging`: 10 instance(s), lines 77, 166, 53, 42, 48, 59, 65, 161, 153, 158

### `scripts/training/extract_structural_nma_v2.py`
- `broad_exception_banned`: 5 instance(s), lines 148, 162, 202, 237, 347
- `print_banned_production_logging`: 9 instance(s), lines 350, 318, 314, 337, 329, 334, 345, 341, 348

### `scripts/training/feature_extractor.py`
- `banned_dataframe_engine`: 1 instance(s), lines 525
- `broad_exception_banned`: 3 instance(s), lines 251, 535, 335
- `print_banned_production_logging`: 11 instance(s), lines 802, 803, 814, 815, 816, 817, 699, 716, 805, 810, 673

### `scripts/training/master_runpod_training.py`
- `broad_exception_banned`: 3 instance(s), lines 491, 275, 498
- `print_banned_production_logging`: 24 instance(s), lines 84, 90, 146, 218, 237, 264, 328, 476, 504, 505, 166, 252, 347, 371, 386, 410, 449, 450, 315, 465, 467, 492, 276, 297

### `scripts/training/phase_manifold_features.py`
- `broad_exception_banned`: 2 instance(s), lines 80, 99
- `direct_parquet_writer_banned`: 1 instance(s), lines 220
- `print_banned_production_logging`: 4 instance(s), lines 221, 78, 81, 216

### `scripts/training/physics_constraints.py`
- `print_banned_production_logging`: 2 instance(s), lines 320, 324

### `scripts/training/physics_predictor.py`
- `print_banned_production_logging`: 23 instance(s), lines 135, 210, 294, 299, 312, 314, 315, 317, 318, 320, 321, 340, 385, 386, 387, 388, 389, 392, 401, 111, 290, 251, 262

### `scripts/training/post_campaign_analysis.py`
- `broad_exception_banned`: 2 instance(s), lines 132, 210
- `json_dump_banned_for_analytical_data`: 2 instance(s), lines 442, 453
- `print_banned_production_logging`: 22 instance(s), lines 310, 406, 423, 424, 439, 455, 456, 457, 460, 462, 464, 465, 466, 399, 401, 410, 448, 459, 147, 417, 419, 438

### `scripts/training/predict_v004.py`
- `print_banned_production_logging`: 19 instance(s), lines 339, 340, 341, 342, 343, 346, 350, 353, 355, 359, 363, 372, 391, 396, 400, 404, 405, 375, 402

### `scripts/training/prism_native_signal_decoders.py`
- `broad_exception_banned`: 3 instance(s), lines 79, 130, 234
- `direct_parquet_writer_banned`: 1 instance(s), lines 272
- `print_banned_production_logging`: 1 instance(s), lines 273

### `scripts/training/prism_only_feature_extractor_v1.py`
- `broad_exception_banned`: 4 instance(s), lines 156, 412, 420, 426
- `direct_parquet_writer_banned`: 1 instance(s), lines 430
- `print_banned_production_logging`: 12 instance(s), lines 200, 201, 203, 205, 233, 356, 364, 432, 350, 413, 421, 427

### `scripts/training/prism_only_feature_extractor_v2.py`
- `broad_exception_banned`: 5 instance(s), lines 109, 184, 315, 323, 329
- `direct_parquet_writer_banned`: 1 instance(s), lines 333
- `print_banned_production_logging`: 10 instance(s), lines 135, 136, 139, 163, 268, 337, 261, 316, 324, 330

### `scripts/training/prism_only_feature_extractor_v3_gpu.py`
- `broad_exception_banned`: 9 instance(s), lines 52, 663, 709, 760, 193, 334, 342, 348, 66
- `direct_parquet_writer_banned`: 2 instance(s), lines 352, 795
- `print_banned_production_logging`: 11 instance(s), lines 152, 153, 158, 287, 356, 796, 155, 280, 335, 343, 349

### `scripts/training/prism_only_feature_extractor_v5.py`
- `banned_text_table_io`: 1 instance(s), lines 21
- `broad_exception_banned`: 11 instance(s), lines 65, 495, 535, 579, 608, 635, 669, 710, 730, 80, 190
- `direct_parquet_writer_banned`: 1 instance(s), lines 822
- `print_banned_production_logging`: 7 instance(s), lines 765, 766, 769, 773, 826, 810, 814

### `scripts/training/site_vqvae.py`
- `print_banned_production_logging`: 18 instance(s), lines 234, 409, 413, 416, 419, 433, 491, 492, 493, 495, 496, 497, 498, 507, 403, 501, 505, 295

### `scripts/training/spike_attention_ranker.py`
- `broad_exception_banned`: 1 instance(s), lines 134
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 701
- `print_banned_production_logging`: 51 instance(s), lines 381, 472, 473, 474, 531, 560, 589, 590, 591, 594, 615, 620, 661, 664, 665, 666, 683, 684, 685, 686, 687, 688, 689, 690, 691, ... (+26 more)

### `scripts/training/spike_bert.py`
- `print_banned_production_logging`: 53 instance(s), lines 88, 106, 120, 320, 377, 403, 411, 414, 415, 416, 436, 437, 438, 446, 454, 455, 456, 459, 476, 477, 478, 497, 498, 504, 505, ... (+28 more)
- `sklearn_model_import_requires_exemption`: 1 instance(s), lines 108

### `scripts/training/spike_bert_v002.py`
- `print_banned_production_logging`: 24 instance(s), lines 48, 57, 63, 227, 250, 256, 274, 275, 278, 295, 331, 332, 336, 357, 369, 378, 379, 380, 381, 92, 284, 285, 328, 354
- `sklearn_model_import_requires_exemption`: 1 instance(s), lines 47

### `scripts/training/spike_bert_v003.py`
- `print_banned_production_logging`: 7 instance(s), lines 307, 327, 406, 219, 315, 390, 403

### `scripts/training/spikebert.py`
- `broad_exception_banned`: 2 instance(s), lines 319, 474
- `print_banned_production_logging`: 18 instance(s), lines 311, 520, 527, 536, 555, 578, 590, 627, 628, 629, 649, 516, 631, 632, 634, 636, 603, 617

### `scripts/training/temporal_tokenizer.py`
- `print_banned_production_logging`: 22 instance(s), lines 308, 309, 329, 337, 355, 376, 377, 384, 387, 388, 389, 390, 391, 392, 393, 394, 395, 397, 398, 302, 319, 340
- `sklearn_model_import_requires_exemption`: 1 instance(s), lines 353

### `scripts/training/train_teacher.py`
- `broad_exception_banned`: 1 instance(s), lines 354
- `print_banned_production_logging`: 26 instance(s), lines 421, 461, 464, 466, 473, 479, 482, 503, 516, 517, 518, 519, 520, 521, 524, 545, 383, 457, 470, 486, 491, 499, 541, 329, 349, ... (+1 more)

### `scripts/training/train_v006.py`
- `broad_exception_banned`: 5 instance(s), lines 424, 157, 501, 186, 208
- `print_banned_production_logging`: 39 instance(s), lines 130, 131, 132, 226, 227, 228, 229, 445, 446, 447, 456, 517, 518, 519, 520, 521, 708, 709, 710, 711, 712, 752, 765, 856, 857, ... (+14 more)

### `scripts/training/validate_176_gate.py`
- `broad_exception_banned`: 1 instance(s), lines 106
- `print_banned_production_logging`: 10 instance(s), lines 95, 154, 155, 156, 159, 160, 162, 167, 158, 166

### `scripts/training/validate_tide_coverage.py`
- `broad_exception_banned`: 1 instance(s), lines 43
- `print_banned_production_logging`: 9 instance(s), lines 84, 85, 86, 87, 88, 89, 90, 93, 92

### `scripts/training/vn_egnn/model.py`
- `print_banned_production_logging`: 5 instance(s), lines 386, 407, 408, 409, 410

### `scripts/training/vn_egnn/train.py`
- `broad_exception_banned`: 1 instance(s), lines 217
- `print_banned_production_logging`: 15 instance(s), lines 277, 281, 285, 308, 323, 371, 378, 393, 398, 273, 287, 290, 349, 373, 361

### `scripts/training/vn_egnn/train_v004.py`
- `print_banned_production_logging`: 51 instance(s), lines 248, 252, 253, 254, 255, 259, 260, 261, 282, 285, 286, 288, 289, 293, 299, 300, 301, 339, 340, 364, 367, 368, 369, 397, 398, ... (+26 more)

### `scripts/training/xgboost_ranker.py`
- `banned_dataframe_engine`: 1 instance(s), lines 58
- `json_dump_banned_for_analytical_data`: 2 instance(s), lines 402, 371
- `print_banned_production_logging`: 31 instance(s), lines 92, 95, 97, 100, 121, 127, 194, 351, 382, 386, 396, 403, 334, 347, 360, 361, 362, 363, 364, 365, 366, 367, 372, 375, 398, ... (+6 more)

### `scripts/training/xgboost_ranker_v2.py`
- `banned_dataframe_engine`: 1 instance(s), lines 43
- `json_dump_banned_for_analytical_data`: 2 instance(s), lines 288, 322
- `print_banned_production_logging`: 27 instance(s), lines 152, 267, 269, 271, 274, 278, 279, 280, 281, 282, 283, 284, 285, 291, 292, 293, 294, 295, 298, 307, 318, 324, 300, 320, 221, ... (+2 more)

### `scripts/training/xgboost_ranker_v3.py`
- `broad_exception_banned`: 3 instance(s), lines 224, 463, 157
- `json_dump_banned_for_analytical_data`: 4 instance(s), lines 492, 510, 545, 219
- `print_banned_production_logging`: 39 instance(s), lines 240, 259, 276, 278, 280, 341, 481, 482, 483, 500, 501, 502, 503, 504, 505, 506, 507, 513, 514, 515, 516, 517, 518, 521, 530, ... (+14 more)

### `scripts/training/xgboost_ranker_v4.py`
- `banned_dataframe_engine`: 1 instance(s), lines 40
- `broad_exception_banned`: 1 instance(s), lines 318
- `json_dump_banned_for_analytical_data`: 2 instance(s), lines 409, 385
- `print_banned_production_logging`: 39 instance(s), lines 89, 92, 96, 100, 116, 189, 357, 396, 405, 411, 412, 413, 261, 313, 343, 351, 355, 376, 377, 378, 379, 380, 381, 382, 383, ... (+14 more)

### `scripts/trajectory_anchor_delta.py`
- `print_banned_production_logging`: 4 instance(s), lines 404, 464, 466, 478

### `scripts/validate_apo_holo.py`
- `print_banned_production_logging`: 8 instance(s), lines 24, 27, 35, 79, 80, 8, 9, 21

### `scripts/validate_chainC_aligned.py`
- `banned_determinant`: 1 instance(s), lines 291
- `broad_exception_banned`: 4 instance(s), lines 52, 61, 102, 154
- `print_banned_production_logging`: 10 instance(s), lines 602, 603, 604, 605, 606, 607, 608, 609, 610, 613

### `scripts/validate_kras_residue_overlap.py`
- `broad_exception_banned`: 3 instance(s), lines 66, 92, 164
- `print_banned_production_logging`: 75 instance(s), lines 265, 279, 280, 281, 282, 283, 284, 285, 286, 291, 300, 317, 318, 319, 320, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, ... (+50 more)

### `scripts/validate_structure.py`
- `broad_exception_banned`: 1 instance(s), lines 517
- `json_dump_banned_for_analytical_data`: 1 instance(s), lines 719
- `print_banned_production_logging`: 49 instance(s), lines 584, 585, 586, 588, 589, 590, 591, 594, 605, 624, 635, 636, 637, 644, 652, 653, 676, 685, 536, 537, 549, 602, 615, 619, 629, ... (+24 more)

### `scripts/validate_topology.py`
- `print_banned_production_logging`: 29 instance(s), lines 291, 292, 293, 294, 295, 296, 297, 298, 304, 314, 324, 334, 344, 352, 359, 377, 306, 308, 316, 318, 326, 328, 336, 338, 346, ... (+4 more)

### `scripts/validation/pfr/pfr_a_vectorial_pharmacophore_gen.py`
- `print_banned_production_logging`: 19 instance(s), lines 85, 86, 487, 488, 489, 490, 491, 517, 518, 519, 520, 464, 504, 523, 429, 434, 502, 511, 512

### `scripts/validation/pfr/pfr_b_temporal_scramble_null.py`
- `print_banned_production_logging`: 12 instance(s), lines 354, 355, 356, 357, 384, 261, 324, 351, 368, 278, 284, 275

### `scripts/validation/pfr/pfr_c_holo_interaction_extractor.py`
- `print_banned_production_logging`: 18 instance(s), lines 633, 634, 635, 636, 637, 638, 668, 669, 670, 671, 672, 577, 593, 613, 659, 573, 645, 646

### `scripts/validation/pfr/pfr_d_vectorial_pfr_scorer.py`
- `banned_determinant`: 1 instance(s), lines 128
- `print_banned_production_logging`: 40 instance(s), lines 153, 702, 703, 704, 705, 706, 707, 708, 709, 755, 756, 757, 758, 760, 761, 762, 763, 767, 768, 769, 770, 773, 774, 780, 781, ... (+15 more)

### `scripts/verify_dossier.py`
- `print_banned_production_logging`: 11 instance(s), lines 11, 12, 13, 19, 20, 27, 67, 44, 49, 56, 61

### `scripts/verify_topology.py`
- `print_banned_production_logging`: 141 instance(s), lines 70, 71, 72, 78, 79, 80, 81, 82, 83, 84, 129, 130, 161, 162, 163, 164, 165, 179, 194, 227, 255, 277, 325, 433, 463, ... (+116 more)

### `scripts/view_topologies.py`
- `print_banned_production_logging`: 7 instance(s), lines 12, 13, 14, 15, 16, 17, 18

### `scripts/viewer/build_viewer.py`
- `print_banned_production_logging`: 1 instance(s), lines 217

### `scripts/visualize_dossier.py`
- `banned_text_table_io`: 1 instance(s), lines 32
- `broad_exception_banned`: 2 instance(s), lines 480, 732
- `print_banned_production_logging`: 5 instance(s), lines 825, 826, 827, 828, 830

### `tests/test_filters/conftest.py`
- `broad_exception_banned`: 1 instance(s), lines 53

### `tests/test_filters/test_filter_pipeline.py`
- `json_dump_banned_for_analytical_data`: 2 instance(s), lines 166, 208

### `tests/test_gating/test_pipeline_integration.py`
- `json_dump_banned_for_analytical_data`: 4 instance(s), lines 112, 124, 222, 231

### `tests/test_genphore/test_spike_to_pharmacophore.py`
- `json_dump_banned_for_analytical_data`: 3 instance(s), lines 70, 276, 292

### `tests/test_interfaces/test_docking_result.py`
- `banned_pickle_serialization`: 1 instance(s), lines 5

### `tests/test_interfaces/test_ensemble_score.py`
- `banned_pickle_serialization`: 1 instance(s), lines 5

### `tests/test_interfaces/test_explicit_solvent_result.py`
- `banned_pickle_serialization`: 1 instance(s), lines 5

### `tests/test_interfaces/test_fep_result.py`
- `banned_pickle_serialization`: 1 instance(s), lines 5

### `tests/test_interfaces/test_filtered_candidate.py`
- `banned_pickle_serialization`: 1 instance(s), lines 5

### `tests/test_interfaces/test_generated_molecule.py`
- `banned_pickle_serialization`: 1 instance(s), lines 5

### `tests/test_interfaces/test_membrane_system.py`
- `banned_pickle_serialization`: 1 instance(s), lines 5

### `tests/test_interfaces/test_pipeline_config.py`
- `banned_pickle_serialization`: 1 instance(s), lines 6

### `tests/test_interfaces/test_pocket_dynamics.py`
- `banned_pickle_serialization`: 1 instance(s), lines 5

### `tests/test_interfaces/test_residue_mapping.py`
- `banned_pickle_serialization`: 1 instance(s), lines 5

### `tests/test_interfaces/test_spike_pharmacophore.py`
- `banned_pickle_serialization`: 1 instance(s), lines 5

### `tests/test_interfaces/test_tautomer_state.py`
- `banned_pickle_serialization`: 1 instance(s), lines 5

### `tests/test_interfaces/test_viewer_payload.py`
- `banned_pickle_serialization`: 1 instance(s), lines 5

### `tests/test_interfaces/test_water_map.py`
- `banned_pickle_serialization`: 1 instance(s), lines 5
