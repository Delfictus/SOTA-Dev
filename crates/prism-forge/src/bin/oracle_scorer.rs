use anyhow::{anyhow, Context, Result};
use arrow_array::{
    Array, ArrayRef, BooleanArray, Float64Array, LargeStringArray, RecordBatch, StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::{arrow_reader::ParquetRecordBatchReaderBuilder, ArrowWriter};
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use serde_json::json;
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::{create_dir_all, File};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

const DEFAULT_BATCH: &str = ".scratch/oracle_batch.parquet";
const DEFAULT_REWARDS: &str = ".scratch/oracle_rewards.parquet";
const DEFAULT_SURVIVORS: &str =
    "campaigns/glp1r_aleniglipron/track_a_generative/vspace_survivors_full_scale.parquet";

#[derive(Debug, Clone)]
struct Config {
    batch: PathBuf,
    rewards: PathBuf,
    survivors: PathBuf,
    lock_mask: Option<PathBuf>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            batch: PathBuf::from(DEFAULT_BATCH),
            rewards: PathBuf::from(DEFAULT_REWARDS),
            survivors: PathBuf::from(DEFAULT_SURVIVORS),
            lock_mask: None,
        }
    }
}

#[derive(Debug, Clone)]
struct SurvivorReward {
    anchor_id: String,
    canonical_smiles: String,
    reward: f64,
    pi_complement: f64,
    adjusted_pi_clash: f64,
    pi_clash_pocket: f64,
    pi_clash_lock: f64,
    pi_clash_lock_per_phase: [f64; 5],
    cryptic_bonus: f64,
    survival_tier: String,
    selected_dihedral_deg: f64,
    lock_proxy_method: String,
}

#[derive(Debug, Clone)]
struct Proposal {
    trajectory_id: String,
    anchor_id: String,
    canonical_smiles: String,
}

#[derive(Debug, Clone)]
struct RewardRow {
    trajectory_id: String,
    anchor_id: String,
    canonical_smiles: String,
    reward: f64,
    pi_complement: f64,
    adjusted_pi_clash: f64,
    pi_clash_pocket: f64,
    pi_clash_lock: f64,
    pi_clash_lock_per_phase: [f64; 5],
    cryptic_bonus: f64,
    survival_tier: String,
    selected_dihedral_deg: f64,
    reward_components_json: String,
    oracle_valid: bool,
}

fn main() -> Result<()> {
    let config = parse_args()?;
    let start = Instant::now();
    let lock_mask = match &config.lock_mask {
        Some(path) => Some(
            LockRegionMask::from_path(path)
                .with_context(|| format!("load lock mask {}", path.display()))?,
        ),
        None => None,
    };
    let survivors = load_survivors(&config.survivors, lock_mask.as_ref())
        .with_context(|| format!("load survivor corpus {}", config.survivors.display()))?;
    let proposals = load_batch(&config.batch)
        .with_context(|| format!("load batch {}", config.batch.display()))?;
    let survivor_by_smiles = best_survivors_by_smiles(survivors);
    let mut reward_rows = Vec::with_capacity(proposals.len());
    for proposal in proposals {
        let Some(reward) = survivor_by_smiles.get(&proposal.canonical_smiles) else {
            reward_rows.push(RewardRow {
                trajectory_id: proposal.trajectory_id,
                anchor_id: proposal.anchor_id,
                canonical_smiles: proposal.canonical_smiles,
                reward: 0.0,
                pi_complement: 0.0,
                adjusted_pi_clash: 0.0,
                pi_clash_pocket: 0.0,
                pi_clash_lock: 0.0,
                pi_clash_lock_per_phase: [0.0; 5],
                cryptic_bonus: 0.0,
                survival_tier: "invalid_missing_survivor".to_owned(),
                selected_dihedral_deg: f64::NAN,
                reward_components_json: json!({
                    "oracle_mode": "validated_o3a_zmatrix_corpus_lookup",
                    "error": "proposal not present in immutable survivor corpus"
                })
                .to_string(),
                oracle_valid: false,
            });
            continue;
        };
        reward_rows.push(RewardRow {
            trajectory_id: proposal.trajectory_id,
            anchor_id: reward.anchor_id.clone(),
            canonical_smiles: reward.canonical_smiles.clone(),
            reward: reward.reward.max(1.0e-8),
            pi_complement: reward.pi_complement,
            adjusted_pi_clash: reward.adjusted_pi_clash,
            pi_clash_pocket: reward.pi_clash_pocket,
            pi_clash_lock: reward.pi_clash_lock,
            pi_clash_lock_per_phase: reward.pi_clash_lock_per_phase,
            cryptic_bonus: reward.cryptic_bonus,
            survival_tier: reward.survival_tier.clone(),
            selected_dihedral_deg: reward.selected_dihedral_deg,
            reward_components_json: json!({
                "oracle_mode": "validated_o3a_zmatrix_corpus_lookup",
                "rust_reward_authority": true,
                "fragment_pi_complement": reward.pi_complement,
                "adjusted_pi_clash": reward.adjusted_pi_clash,
                "pi_clash_pocket": reward.pi_clash_pocket,
                "pi_clash_lock": reward.pi_clash_lock,
                "pi_clash_lock_cold_hold": reward.pi_clash_lock_per_phase[0],
                "pi_clash_lock_ramp_up": reward.pi_clash_lock_per_phase[1],
                "pi_clash_lock_warm_hold": reward.pi_clash_lock_per_phase[2],
                "pi_clash_lock_ramp_down": reward.pi_clash_lock_per_phase[3],
                "pi_clash_lock_cold_return": reward.pi_clash_lock_per_phase[4],
                "lock_proxy_method": reward.lock_proxy_method,
                "cryptic_bonus": reward.cryptic_bonus,
                "survival_tier": reward.survival_tier,
                "selected_dihedral_deg": reward.selected_dihedral_deg
            })
            .to_string(),
            oracle_valid: true,
        });
    }
    write_rewards(&config.rewards, &reward_rows)?;
    let invalid = reward_rows.iter().filter(|row| !row.oracle_valid).count();
    let mean_reward = if reward_rows.is_empty() {
        0.0
    } else {
        reward_rows.iter().map(|row| row.reward).sum::<f64>() / reward_rows.len() as f64
    };
    println!(
        "oracle_scorer batch={} valid={} invalid={} reward_mean={:.6} elapsed_ms={:.3} rewards={}",
        reward_rows.len(),
        reward_rows.len().saturating_sub(invalid),
        invalid,
        mean_reward,
        start.elapsed().as_secs_f64() * 1000.0,
        config.rewards.display()
    );
    Ok(())
}

fn parse_args() -> Result<Config> {
    let mut config = Config::default();
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--batch" => config.batch = PathBuf::from(value_after(&mut args, "--batch")?),
            "--rewards" => config.rewards = PathBuf::from(value_after(&mut args, "--rewards")?),
            "--survivors" => {
                config.survivors = PathBuf::from(value_after(&mut args, "--survivors")?)
            }
            "--lock-mask" => {
                config.lock_mask = Some(PathBuf::from(value_after(&mut args, "--lock-mask")?))
            }
            "--help" | "-h" => {
                println!(
                    "oracle_scorer --batch <parquet> --rewards <parquet> --survivors <parquet> [--lock-mask <json>]"
                );
                std::process::exit(0);
            }
            _ => return Err(anyhow!("unknown argument {arg}")),
        }
    }
    Ok(config)
}

fn value_after(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String> {
    args.next()
        .ok_or_else(|| anyhow!("{flag} requires a value"))
}

fn best_survivors_by_smiles(survivors: Vec<SurvivorReward>) -> HashMap<String, SurvivorReward> {
    let mut by_smiles: HashMap<String, SurvivorReward> = HashMap::new();
    for survivor in survivors {
        by_smiles
            .entry(survivor.canonical_smiles.clone())
            .and_modify(|existing| {
                if survivor.reward > existing.reward {
                    *existing = survivor.clone();
                }
            })
            .or_insert(survivor);
    }
    by_smiles
}

fn load_batch(path: &Path) -> Result<Vec<Proposal>> {
    let mut proposals = Vec::new();
    for batch in read_batches(path)? {
        let trajectory = string_column(&batch, "trajectory_id")?;
        let anchors = string_column(&batch, "anchor_id")?;
        let smiles = string_column(&batch, "canonical_smiles")?;
        for row_idx in 0..batch.num_rows() {
            proposals.push(Proposal {
                trajectory_id: string_value(trajectory, row_idx)?,
                anchor_id: string_value(anchors, row_idx)?,
                canonical_smiles: string_value(smiles, row_idx)?,
            });
        }
    }
    Ok(proposals)
}

fn load_survivors(path: &Path, lock_mask: Option<&LockRegionMask>) -> Result<Vec<SurvivorReward>> {
    let mut survivors = Vec::new();
    for batch in read_batches(path)? {
        let anchors = string_column(&batch, "anchor_id")?;
        let smiles = string_column(&batch, "canonical_smiles")?;
        let complements = f64_column(&batch, "fragment_pi_complement")?;
        let clashes = f64_column(&batch, "fragment_pi_clash_adjusted")?;
        let cryptic = f64_column(&batch, "cryptic_bonus")?;
        let tiers = string_column(&batch, "survival_tier")?;
        let dihedrals = f64_column(&batch, "selected_dihedral_deg")?;
        let coordinates = optional_string_column(&batch, "coordinates_json");
        for row_idx in 0..batch.num_rows() {
            let adjusted_pi_clash = f64_value(clashes, row_idx)?;
            let coordinates_json = coordinates
                .as_ref()
                .and_then(|array| string_value(*array, row_idx).ok())
                .unwrap_or_default();
            let lock_proxy = bifurcate_clash(&coordinates_json, adjusted_pi_clash, lock_mask);
            survivors.push(SurvivorReward {
                anchor_id: string_value(anchors, row_idx)?,
                canonical_smiles: string_value(smiles, row_idx)?,
                reward: bifurcated_reward(
                    f64_value(complements, row_idx)?,
                    lock_proxy.pi_clash_pocket,
                    lock_proxy.pi_clash_lock,
                    f64_value(cryptic, row_idx)?,
                ),
                pi_complement: f64_value(complements, row_idx)?,
                adjusted_pi_clash,
                pi_clash_pocket: lock_proxy.pi_clash_pocket,
                pi_clash_lock: lock_proxy.pi_clash_lock,
                pi_clash_lock_per_phase: lock_proxy.pi_clash_lock_per_phase,
                cryptic_bonus: f64_value(cryptic, row_idx)?,
                survival_tier: string_value(tiers, row_idx)?,
                selected_dihedral_deg: f64_value(dihedrals, row_idx)?,
                lock_proxy_method: lock_proxy.method,
            });
        }
    }
    Ok(survivors)
}

#[derive(Debug, Clone)]
struct ClashBifurcation {
    pi_clash_pocket: f64,
    pi_clash_lock: f64,
    pi_clash_lock_per_phase: [f64; 5],
    method: String,
}

#[derive(Debug, Clone)]
struct LockRegionMask {
    lock_voxels: HashSet<u64>,
    origin: [f64; 3],
    spacing: f64,
    dims: [u64; 3],
}

impl LockRegionMask {
    fn from_path(path: &Path) -> Result<Self> {
        let decoded: serde_json::Value = serde_json::from_reader(File::open(path)?)?;
        let lock_voxels = decoded
            .get("lock_voxel_indices")
            .and_then(|value| value.as_array())
            .ok_or_else(|| anyhow!("lock mask missing lock_voxel_indices"))?
            .iter()
            .filter_map(|value| value.as_u64())
            .collect::<HashSet<_>>();
        let grid = decoded
            .get("grid")
            .ok_or_else(|| anyhow!("lock mask missing grid object"))?;
        let origin_values = grid
            .get("origin_xyz_angstrom")
            .and_then(|value| value.as_array())
            .ok_or_else(|| anyhow!("lock mask grid missing origin_xyz_angstrom"))?;
        let origin = [
            origin_values.first().and_then(|value| value.as_f64()).unwrap_or(0.0),
            origin_values.get(1).and_then(|value| value.as_f64()).unwrap_or(0.0),
            origin_values.get(2).and_then(|value| value.as_f64()).unwrap_or(0.0),
        ];
        let dims = [
            grid.get("nx").and_then(|value| value.as_u64()).unwrap_or(96),
            grid.get("ny").and_then(|value| value.as_u64()).unwrap_or(96),
            grid.get("nz").and_then(|value| value.as_u64()).unwrap_or(96),
        ];
        let spacing = grid
            .get("spacing_angstrom")
            .and_then(|value| value.as_f64())
            .ok_or_else(|| anyhow!("lock mask grid missing spacing_angstrom"))?;
        Ok(Self {
            lock_voxels,
            origin,
            spacing,
            dims,
        })
    }

    fn contains_xyz(&self, xyz: &[f64]) -> bool {
        if xyz.len() < 3 || self.spacing <= 0.0 {
            return false;
        }
        let ix = ((xyz[0] - self.origin[0]) / self.spacing).floor() as i64;
        let iy = ((xyz[1] - self.origin[1]) / self.spacing).floor() as i64;
        let iz = ((xyz[2] - self.origin[2]) / self.spacing).floor() as i64;
        if ix < 0
            || iy < 0
            || iz < 0
            || ix >= self.dims[0] as i64
            || iy >= self.dims[1] as i64
            || iz >= self.dims[2] as i64
        {
            return false;
        }
        let voxel_idx = iz as u64 * (self.dims[0] * self.dims[1]) + iy as u64 * self.dims[0] + ix as u64;
        self.lock_voxels.contains(&voxel_idx)
    }
}

fn bifurcated_reward(
    pi_complement: f64,
    pi_clash_pocket: f64,
    pi_clash_lock: f64,
    cryptic_bonus: f64,
) -> f64 {
    (pi_complement + cryptic_bonus + pi_clash_lock - pi_clash_pocket).max(1.0e-8)
}

fn bifurcate_clash(
    coordinates_json: &str,
    adjusted_pi_clash: f64,
    lock_mask: Option<&LockRegionMask>,
) -> ClashBifurcation {
    let parsed = serde_json::from_str::<Vec<Vec<f64>>>(coordinates_json);
    let Ok(coords) = parsed else {
        return ClashBifurcation {
            pi_clash_pocket: adjusted_pi_clash,
            pi_clash_lock: 0.0,
            pi_clash_lock_per_phase: [0.0; 5],
            method: "missing_coordinates_fallback_all_pocket".to_owned(),
        };
    };
    if let Some(mask) = lock_mask {
        let lock_atom_count = coords.iter().filter(|coord| mask.contains_xyz(coord)).count();
        let atom_count = coords.len().max(1);
        let lock_fraction = lock_atom_count as f64 / atom_count as f64;
        let pi_clash_lock = adjusted_pi_clash * lock_atom_count as f64;
        return ClashBifurcation {
            pi_clash_pocket: adjusted_pi_clash * (1.0 - lock_fraction),
            pi_clash_lock,
            pi_clash_lock_per_phase: [pi_clash_lock; 5],
            method: format!(
                "residue_lock_region_mask_v1:lock_atoms={lock_atom_count}:total_atoms={atom_count}:phase_lock=aggregate_replicated"
            ),
        };
    }
    let z_values: Vec<f64> = coords
        .iter()
        .filter_map(|coord| coord.get(2).copied())
        .filter(|value| value.is_finite())
        .collect();
    if z_values.is_empty() {
        return ClashBifurcation {
            pi_clash_pocket: adjusted_pi_clash,
            pi_clash_lock: 0.0,
            pi_clash_lock_per_phase: [0.0; 5],
            method: "empty_coordinates_fallback_all_pocket".to_owned(),
        };
    }
    let z_min = z_values.iter().copied().fold(f64::INFINITY, f64::min);
    let z_max = z_values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let z_span = (z_max - z_min).max(1.0e-9);
    let lock_cutoff = z_min + (0.20 * z_span);
    let lock_atom_count = z_values.iter().filter(|z| **z <= lock_cutoff).count();
    let atom_count = z_values.len().max(1);
    let lock_fraction = lock_atom_count as f64 / atom_count as f64;
    let pi_clash_lock = adjusted_pi_clash * lock_atom_count as f64;
    ClashBifurcation {
        pi_clash_pocket: adjusted_pi_clash * (1.0 - lock_fraction),
        pi_clash_lock,
        pi_clash_lock_per_phase: [pi_clash_lock; 5],
        method: format!(
            "intracellular_bottom_z_proxy_v1:z_min={z_min:.4}:z_max={z_max:.4}:cutoff={lock_cutoff:.4}:lock_atoms={lock_atom_count}:total_atoms={atom_count}:phase_lock=aggregate_replicated"
        ),
    }
}

fn read_batches(path: &Path) -> Result<Vec<RecordBatch>> {
    let file = File::open(path)?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;
    reader
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(Into::into)
}

fn column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a dyn Array> {
    let idx = batch
        .schema()
        .index_of(name)
        .with_context(|| format!("missing column {name}"))?;
    Ok(batch.column(idx).as_ref())
}

fn string_column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a dyn Array> {
    let array = column(batch, name)?;
    if array.as_any().downcast_ref::<StringArray>().is_some()
        || array.as_any().downcast_ref::<LargeStringArray>().is_some()
    {
        Ok(array)
    } else {
        Err(anyhow!("column {name} is not string"))
    }
}

fn optional_string_column<'a>(batch: &'a RecordBatch, name: &str) -> Option<&'a dyn Array> {
    let idx = batch.schema().index_of(name).ok()?;
    let array = batch.column(idx).as_ref();
    if array.as_any().downcast_ref::<StringArray>().is_some()
        || array.as_any().downcast_ref::<LargeStringArray>().is_some()
    {
        Some(array)
    } else {
        None
    }
}

fn f64_column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Float64Array> {
    column(batch, name)?
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| anyhow!("column {name} is not Float64"))
}

fn string_value(array: &dyn Array, row_idx: usize) -> Result<String> {
    if array.is_null(row_idx) {
        return Err(anyhow!("null string at row {row_idx}"));
    }
    if let Some(values) = array.as_any().downcast_ref::<StringArray>() {
        return Ok(values.value(row_idx).to_owned());
    }
    if let Some(values) = array.as_any().downcast_ref::<LargeStringArray>() {
        return Ok(values.value(row_idx).to_owned());
    }
    Err(anyhow!("array is not string"))
}

fn f64_value(array: &Float64Array, row_idx: usize) -> Result<f64> {
    if array.is_null(row_idx) {
        return Err(anyhow!("null f64 at row {row_idx}"));
    }
    Ok(array.value(row_idx))
}

fn write_rewards(path: &Path, rows: &[RewardRow]) -> Result<()> {
    if let Some(parent) = path.parent() {
        create_dir_all(parent)?;
    }
    let schema = Arc::new(Schema::new(vec![
        Field::new("trajectory_id", DataType::Utf8, false),
        Field::new("anchor_id", DataType::Utf8, false),
        Field::new("canonical_smiles", DataType::Utf8, false),
        Field::new("reward", DataType::Float64, false),
        Field::new("pi_complement", DataType::Float64, false),
        Field::new("adjusted_pi_clash", DataType::Float64, false),
        Field::new("pi_clash_pocket", DataType::Float64, false),
        Field::new("pi_clash_lock", DataType::Float64, false),
        Field::new("pi_clash_lock_cold_hold", DataType::Float64, false),
        Field::new("pi_clash_lock_ramp_up", DataType::Float64, false),
        Field::new("pi_clash_lock_warm_hold", DataType::Float64, false),
        Field::new("pi_clash_lock_ramp_down", DataType::Float64, false),
        Field::new("pi_clash_lock_cold_return", DataType::Float64, false),
        Field::new("cryptic_bonus", DataType::Float64, false),
        Field::new("survival_tier", DataType::Utf8, false),
        Field::new("selected_dihedral_deg", DataType::Float64, false),
        Field::new("reward_components_json", DataType::Utf8, false),
        Field::new("oracle_valid", DataType::Boolean, false),
    ]));
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(
            rows.iter()
                .map(|row| row.trajectory_id.as_str())
                .collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter()
                .map(|row| row.anchor_id.as_str())
                .collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter()
                .map(|row| row.canonical_smiles.as_str())
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.reward).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.pi_complement).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter()
                .map(|row| row.adjusted_pi_clash)
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.pi_clash_pocket).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.pi_clash_lock).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter()
                .map(|row| row.pi_clash_lock_per_phase[0])
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter()
                .map(|row| row.pi_clash_lock_per_phase[1])
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter()
                .map(|row| row.pi_clash_lock_per_phase[2])
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter()
                .map(|row| row.pi_clash_lock_per_phase[3])
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter()
                .map(|row| row.pi_clash_lock_per_phase[4])
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.cryptic_bonus).collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter()
                .map(|row| row.survival_tier.as_str())
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter()
                .map(|row| row.selected_dihedral_deg)
                .collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter()
                .map(|row| row.reward_components_json.as_str())
                .collect::<Vec<_>>(),
        )),
        Arc::new(BooleanArray::from(
            rows.iter().map(|row| row.oracle_valid).collect::<Vec<_>>(),
        )),
    ];
    let batch = RecordBatch::try_new(schema.clone(), arrays)?;
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .build();
    let file = File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}
