# Hydration V2 Full Latest

status: FULL_EXTRACTION_RUN_AND_VALIDATED

run_root: `campaigns/glp1r_aleniglipron/integrated_spike_events/hydration_v2_full_20260529T090620Z`

input: `campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/spike_events_snr_masked.parquet`

core_output: `campaigns/glp1r_aleniglipron/integrated_spike_events/hydration_v2_full_20260529T090620Z/hydration_statistics.parquet`

continuity_output: `campaigns/glp1r_aleniglipron/integrated_spike_events/hydration_v2_full_20260529T090620Z/continuity_maps_full/hydration_continuity_map.parquet`

validation:
- extraction exit status: 0
- full hydration rows: 963605
- hydration continuity rows: 963605
- blocked hydration rows: 0
- captured graph spectral linked rows: 4460
- topology region linked rows: 61174
- unmapped rows: 897971
- output sha256: `6dd51bc7e645f7a946b49ba6ab2f55a16320f203f6ba546c77f187c9c1c076a6`

evidence:
- `hydration_v2_full_20260529T090620Z/HYDRATION_FULL_RUN_SUMMARY.json`
- `hydration_v2_full_20260529T090620Z/HYDRATION_FULL_RUN_SUMMARY.md`
- `hydration_v2_full_20260529T090620Z/HYDRATION_FULL_VALIDATION.json`
- `hydration_v2_full_20260529T090620Z/continuity_maps_full/continuity_map_manifest.json`
