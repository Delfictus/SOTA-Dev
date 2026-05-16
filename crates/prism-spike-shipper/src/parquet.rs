use crate::schema::{derive_mechanism_tag, SpikeFile};
use anyhow::{Context, Result};
use arrow_array::{
    builder::Float32Builder, ArrayRef, Float32Array, Int32Array, StringArray, UInt32Array,
};
use arrow_schema::{DataType, Field, Schema};
use parquet::{
    arrow::ArrowWriter,
    basic::{Compression, ZstdLevel},
    file::properties::WriterProperties,
};
use std::{fs::File, path::Path, sync::Arc};

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        // ── required ─────────────────────────────────────────────
        Field::new("timestep", DataType::Int32, false),
        Field::new("frame_index", DataType::Int32, false),
        Field::new("site_id", DataType::Int32, false),
        Field::new("spike_source", DataType::Utf8, false),
        Field::new("ccns_phase", DataType::Utf8, false),
        Field::new("intensity", DataType::Float32, false),
        Field::new("vibrational_energy", DataType::Float32, false),
        Field::new("n_nearby_excited", DataType::Int32, false),
        Field::new("aromatic_residue_id", DataType::Int32, false),
        Field::new("stream_id", DataType::Int32, false),
        Field::new("type", DataType::Utf8, false),
        Field::new("mechanism_tag", DataType::Utf8, false),
        Field::new("phase_bits", DataType::UInt32, false),
        Field::new("wd_change", DataType::Float32, false),
        Field::new("x", DataType::Float32, false),
        Field::new("y", DataType::Float32, false),
        Field::new("z", DataType::Float32, false),
        // ── nullable optional ────────────────────────────────────
        Field::new("water_density", DataType::Float32, true),
        Field::new("wavelength_nm", DataType::Float32, true),
    ]))
}

/// Write spike data to a zstd-compressed Parquet file. Returns row count.
/// Column names are identical to the Python prism_spike_watcher.py schema
/// so feature_extractor.py Block 6 can read both sources interchangeably.
pub fn write(spike_file: &SpikeFile, out: &Path) -> Result<usize> {
    let spikes = &spike_file.spikes;
    let n = spikes.len();
    let sid = spike_file.site_id as i32;

    macro_rules! col_i32 {
        ($e:expr) => {
            Arc::new(Int32Array::from_iter_values($e)) as ArrayRef
        };
    }
    macro_rules! col_u32 {
        ($e:expr) => {
            Arc::new(UInt32Array::from_iter_values($e)) as ArrayRef
        };
    }
    macro_rules! col_f32 {
        ($e:expr) => {
            Arc::new(Float32Array::from_iter_values($e)) as ArrayRef
        };
    }
    macro_rules! col_str {
        ($e:expr) => {
            Arc::new(StringArray::from_iter_values($e)) as ArrayRef
        };
    }
    macro_rules! col_f32_opt {
        ($field:ident) => {{
            let mut b = Float32Builder::with_capacity(n);
            for s in spikes {
                match s.$field {
                    Some(v) => b.append_value(v),
                    None => b.append_null(),
                }
            }
            Arc::new(b.finish()) as ArrayRef
        }};
    }

    let schema = schema();
    let batch = arrow_array::RecordBatch::try_new(
        schema.clone(),
        vec![
            col_i32!(spikes.iter().map(|s| s.timestep)),
            col_i32!(spikes.iter().map(|s| s.frame_index.unwrap_or(0))),
            col_i32!(std::iter::repeat(sid).take(n)),
            col_str!(spikes
                .iter()
                .map(|s| s.spike_source.as_deref().unwrap_or(""))),
            col_str!(spikes.iter().map(|s| s.ccns_phase.as_str())),
            col_f32!(spikes.iter().map(|s| s.intensity)),
            col_f32!(spikes.iter().map(|s| s.vibrational_energy.unwrap_or(0.0))),
            col_i32!(spikes.iter().map(|s| s.n_nearby_excited.unwrap_or(0))),
            col_i32!(spikes.iter().map(|s| s.aromatic_residue_id.unwrap_or(0))),
            col_i32!(spikes.iter().map(|s| s.stream_id.unwrap_or(0))),
            col_str!(spikes.iter().map(|s| s.spike_type.as_deref().unwrap_or(""))),
            col_str!(spikes.iter().map(|s| s
                .mechanism_tag
                .as_deref()
                .unwrap_or_else(|| derive_mechanism_tag(s)))),
            col_u32!(spikes.iter().map(|s| s.phase_bits.unwrap_or(0))),
            col_f32!(spikes.iter().map(|s| s.wd_change.unwrap_or(0.0))),
            // x/y/z are required in SpikeEvent (validated upstream, always finite)
            col_f32!(spikes.iter().map(|s| s.x)),
            col_f32!(spikes.iter().map(|s| s.y)),
            col_f32!(spikes.iter().map(|s| s.z)),
            // optional
            col_f32_opt!(water_density),
            col_f32_opt!(wavelength_nm),
        ],
    )
    .context("RecordBatch::try_new")?;

    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(3).unwrap()))
        .set_dictionary_enabled(true)
        .build();

    let file = File::create(out).with_context(|| format!("create {}", out.display()))?;
    let mut wr = ArrowWriter::try_new(file, schema, Some(props)).context("ArrowWriter::try_new")?;
    wr.write(&batch).context("ArrowWriter::write")?;
    wr.close().context("ArrowWriter::close")?;

    Ok(n)
}
