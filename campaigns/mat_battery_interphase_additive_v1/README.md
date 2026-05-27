# PRISM Materials Track: Battery Interphase Additive v1

This campaign directory anchors the Epoch 025 hardened release Materials
subsystem in the CBOM. The release validates the Materials runtime interface:

- `BatteryInterphaseReward` computes finite nonzero SEI reward values from
  electronic, mechanical, ion, and pose inputs.
- `CCNS_TO_BATTERY_PHASE` maps the five PRISM phases onto charge/discharge
  battery-cycle semantics.
- `UniversalMaterialsActionSpace` instantiates deterministic edit actions for
  organic, inorganic, surface, and coordination chemistry edits.
- `XTBRewardAdapter` exposes HOMO-LUMO and electron-affinity parsing interfaces
  without requiring an xTB binary for CI smoke validation.
- `prism-nhs` compiles the `warp_jacobian` binary used by mechanical/shear
  materials scoring surfaces.

No experimental battery efficacy claim is made by this release.
