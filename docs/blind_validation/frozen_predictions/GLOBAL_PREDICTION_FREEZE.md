# PRISM4D Global Prediction Freeze

Freeze commit: c8b09678

Freeze scope:
- B01_HRAS_Q61H
- B02_CDK2_allosteric
- B03_Kv1.2
- B04_MDM2
- B05_TP53_R175H
- B06_cGAS
- B07_TEAD1
- B08_CRBN
- B09_Thrombin_exosite
- B10_ADRB2

Rule:
PRISM4D candidate ranks, baseline outputs, and scoring definitions were frozen before post-freeze holo-coordinate validation. Post-freeze manifold, thermodynamic, CCNS, hysteresis, TIDE, and KCC descriptors are mechanistic annotations only and were not used to alter frozen primary rank order or primary Strict SR@k.

B10 ADRB2 is retained separately as a hard-negative/stress-test calibration target and is excluded from the 9-target primary macro average.
