# PRISM GLP-1R M2 v1.1 — Known Limitations

These limitations are part of the delivery itself, not concessions.
Every downstream citation must remain consistent with them.

## Derivation chain

- The PDF layer is a *derived* artifact. Its source-of-truth is the
  verified Markdown / Parquet / JSON content in `05_GROUND_TRUTH_DATA/`,
  `07_AUDIT_AND_CBOM/`, and `08_RELEASE_ARCHIVES/`. If a PDF and the
  ground-truth disagree, the ground-truth wins.
- No claims have been added by the PDF layer. The PDF formatter
  may re-order or section the content, but it does not introduce
  new biological assertions.

## Phase 2D — staged, not executed

- The Phase 2D variant grid manifest in this package is in
  `materialization_status: staged`. The engine has not yet been run
  on those targets in the v1.1 timeline. The rows in
  `09_TABLE_EXPORTS/Phase2D_Staged_Targets.csv` are a planning queue,
  not an evidence set.

## Zero-shot replacements — PROJECTED / HYPOTHESIZED

- The Top-10 replacements in `ZeroShot_Top10_Replacements` are
  computational projections from the manual emulation track.
- They are **not validated compounds**, **not synthesis instructions**,
  and **not biological recommendations**.
- They are SAR-contingency shortlist inputs subject to medicinal-
  chemistry review and wet-lab falsification.

## CRO action plan — falsification gates only

- Every row of `CRO_WetLab_Action_Plan` is a falsification gate.
- The associated PRISM-4D claim is at risk if and only if the gate
  fails as described in its `falsification_condition` field.
- Priority score is a routing weight, not a probability of success.

## No clinical / patient claims

- Nothing in this package is a clinical-effect claim.
- Nothing in this package is a patient-response prediction.
- Nothing in this package is experimental validation.

## Scope separation

- This is the lightweight executive delivery. The full raw audit
  archive (large, including raw spike events, full mechanical-load
  networks, and raw `.bin` files) is intentionally **not** included
  in this delivery tarball. It remains separately stored and is
  available on request through the campaign's data-room procedure.

## Visualizer epistemic overlays

- The visualizer color-codes PROJECTED / HYPOTHESIZED layers so that
  falsification gates are reviewable. Visibility in the viewer is
  not a citation license — it is a review affordance.
