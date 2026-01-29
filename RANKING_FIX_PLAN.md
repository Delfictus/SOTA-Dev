# Fixing Site Ranking for Accurate Cryptic Binding Site Detection

## Problem Diagnosis

**Current Performance** (PTP1B example):
- Sites ARE detected (best F1 = 0.304)
- But ranking is WRONG (Hit@1 = false, Hit@3 = false)
- All sites have similar scores (0.645-0.698)
- True binding site buried at rank >3

**Root Causes**:

1. ❌ **Persistence all 0.010** (1%) - Too uniform, can't discriminate
2. ❌ **UV delta SASA is global** - Not site-specific aromatic enrichment
3. ❌ **No aromatic clustering metric** - UV-validated sites not boosted
4. ❌ **Replica agreement same for all** - Not properly computed

---

## Solution: Multi-Tiered Scoring with UV-LIF Physics

### Core Principle

**UV validation ASSISTS core MD physics** (not replaces):
- **Primary signal**: Dewetting events (hydrophobic exclusion + persistence)
- **UV boost**: Sites with high aromatic enrichment get confidence bonus
- **Final ranking**: Dewetting strength × UV confidence × druggability

### New Scoring Formula

```rust
// TIER 1: Core dewetting physics (50% weight)
let dewetting_score =
    0.4 * persistence_fraction +  // How often site appears
    0.3 * event_density +          // Events per Ų
    0.3 * hydrophobic_lining;      // Pocket character

// TIER 2: UV-LIF validation (30% weight)
let uv_confidence_score =
    0.5 * aromatic_enrichment +    // UV spikes vs thermal baseline per site
    0.3 * aromatic_clustering +    // % of site residues that are aromatic
    0.2 * uv_response_strength;    // Total UV spike count in site

// TIER 3: Druggability (20% weight)
let druggability_score =
    0.4 * volume_score +           // 500-2000 Ų optimal
    0.3 * depth_score +            // Pocket depth
    0.3 * shape_score;             // Sphericity, mouth area

// FINAL SCORE
rank_score = 0.50 * dewetting_score
           + 0.30 * uv_confidence_score
           + 0.20 * druggability_score
```

---

## Implementation Changes Needed

### 1. Compute Per-Site Aromatic Enrichment

**Current**: UV delta SASA (global ablation comparison)
**New**: Aromatic enrichment from spike events per site

```rust
fn compute_site_aromatic_enrichment(
    site_residues: &[i32],
    spike_events: &[GpuSpikeEvent],
    aromatic_ids: &HashSet<i32>,
    uv_burst_interval: i32,
    uv_burst_duration: i32,
) -> f32 {
    // Filter spikes in this site's spatial region
    let site_spikes: Vec<_> = spike_events.iter()
        .filter(|spike| {
            let n = spike.n_residues as usize;
            (0..n).any(|i| site_residues.contains(&spike.nearby_residues[i]))
        })
        .collect();

    // Separate UV-phase vs non-UV-phase
    let uv_spikes: Vec<_> = site_spikes.iter()
        .filter(|s| (s.timestep % uv_burst_interval) < uv_burst_duration)
        .collect();
    let non_uv_spikes: Vec<_> = site_spikes.iter()
        .filter(|s| (s.timestep % uv_burst_interval) >= uv_burst_duration)
        .collect();

    // Count aromatic hits
    let uv_with_aro = uv_spikes.iter().filter(|s| {
        let n = s.n_residues as usize;
        (0..n).any(|i| aromatic_ids.contains(&s.nearby_residues[i]))
    }).count();

    let non_uv_with_aro = non_uv_spikes.iter().filter(|s| {
        let n = s.n_residues as usize;
        (0..n).any(|i| aromatic_ids.contains(&s.nearby_residues[i]))
    }).count();

    // Enrichment ratio
    let uv_rate = if !uv_spikes.is_empty() {
        uv_with_aro as f32 / uv_spikes.len() as f32
    } else { 0.0 };

    let non_uv_rate = if !non_uv_spikes.is_empty() {
        non_uv_with_aro as f32 / non_uv_spikes.len() as f32
    } else { 0.01 };  // Avoid division by zero

    let enrichment = uv_rate / non_uv_rate;

    // Normalize: >2.0x = 1.0, <1.0x = 0.0
    ((enrichment - 1.0) / 1.0).clamp(0.0, 1.0)
}
```

### 2. Compute Event Density Per Site

**Current**: Not computed
**New**: Events per cubic angstrom (indicates strong dewetting)

```rust
let event_density = site_event_count as f64 / site_volume.max(100.0);
let density_score = (event_density / 10.0).min(1.0);  // 10 events/Ų = 1.0
```

### 3. Fix Persistence Calculation

**Current**: All sites showing 0.010 (broken)
**Check**: Are we computing persistence correctly across frames?

**Expected**: Sites should have 0.05-0.80 range (5%-80% of frames)

### 4. Boost Aromatic-Rich Sites

```rust
let aromatic_fraction = site.residues.iter()
    .filter(|r| is_aromatic(r))
    .count() as f32 / site.residues.len() as f32;

let aromatic_clustering_score = aromatic_fraction;  // 0-1
```

---

## Immediate Action Items

### Critical Path (Blocking Client Demos)

1. ✅ **Add per-site aromatic enrichment** to `SiteMetrics`
2. ✅ **Update ranking formula** to use enrichment instead of delta SASA
3. ✅ **Fix persistence calculation** (shouldn't all be 0.010)
4. ✅ **Add event density** metric
5. ✅ **Re-run PTP1B** and verify Hit@1, Hit@3 improve

### Quality Assurance

6. ✅ **Verify prism-prep** is producing correct topologies (charges, protonation)
7. ✅ **Check if ablation is needed** or if we can use spike enrichment directly
8. ✅ **Test on 5 structures** with known binding sites
9. ✅ **Measure real Precision@10, Hit@1, Hit@3**

---

## Expected Improvement

**Before (current)**:
```
Hit@1:  0/5 (0.0%)
Hit@3:  1/5 (20.0%)
Precision@10: Unknown
```

**After (with aromatic enrichment ranking)**:
```
Hit@1:  3/5 (60%)   ← Target: competitive with SiteMap
Hit@3:  4/5 (80%)   ← Target: match industry standards
Precision@10: 70%+  ← Sites with high aromatic enrichment ranked higher
```

---

## Integration with Core Physics

The ranking will prioritize sites that have:

✅ **Strong core physics** (dewetting, persistence, hydrophobic exclusion)
✅ **UV-LIF validation** (aromatic enrichment >1.5x proves it's real)
✅ **Druggable geometry** (volume, depth, shape)

This ensures:
- UV validation ASSISTS (not replaces) core MD physics
- Authentic cryptic binding sites (not just aromatic clusters)
- Sites ranked by genuine binding potential (dewetting + dynamics + validation)

---

## Prism-Prep Quality Check

**Current assumption**: Topologies are correct
**Need to verify**:
- ✅ Protonation states (His, Asp/Glu, Lys/Arg at pH 7)
- ✅ Disulfide bonds identified
- ✅ Missing residues handled
- ✅ Charges correct (net charge reasonable)
- ✅ AMBER FF parameters assigned

**Check**: Run prism-prep on test structure and inspect topology.json validation output

---

**Next Step**: Should I implement the per-site aromatic enrichment calculation and update the ranking formula?
