# Track B Variant Physics Auditor

Verdict: PASS.

Runtime evidence:
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_genealogical_variant_panel.py` -> `4 passed in 0.02s`.
- Independent panel scan:
  - `variant_count=96`
  - all eight perturbation families represented, `12` variants each
  - all six topology regions represented, `16` variants each
  - `conservation_used_count=72`
  - exact conservation AA checks: `16`, mismatches: `0`
  - bad residue IDs/source mismatches/generic X/no-op/conservation-only selections: `0`

Code-path evidence:
- `scripts/generate_genealogical_variant_panel.py` parses AA3 identities including protonation aliases.
- Non-sequence selection features are always included before conservation is added.
- Generic/unknown residue identities are filtered before variant generation.
- No-op substitutions and conservation-only variants are rejected.
- Generated variant records include source AA, target AA, topology region, perturbation family, observability channels, and selection features.

Artifact hashes:
- `genealogical_variant_panel.json`: `05a352806bd48f29c2735507e4ad28f7d4d779e822a9dd1a80482e6edc99ffff`
- `topology_region_registry.json`: `d83dc837e99461ee62c55f9c96158d7d06af21e234230bb6ba6713f952cbd20c`
- `GLP1R_cross_species_conservation.csv`: `c2686e9efdb62b41ea5f0f675c0577020565fba1a9d8b445bd23e90589694aff`

Findings:
- CRITICAL: none.
- HIGH: none.
- MEDIUM: none.
- LOW: conservation is exact for only `16` variants and nearest/annotation for the rest. This is acceptable because source amino acid authority comes from registry residue IDs.
