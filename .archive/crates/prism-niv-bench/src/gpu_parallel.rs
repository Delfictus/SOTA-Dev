use anyhow::{Context, Result};
use cudarc::driver::{CudaStream, LaunchConfig, CudaContext, CudaSlice};
use std::sync::Arc;
use std::path::Path;
use std::collections::HashMap;

// Use prism-gpu exports directly
use prism_gpu::{MegaFusedBatchGpu, MegaFusedConfig};
use crate::{PackedBatch};  // Re-exported from prism-gpu via lib.rs
// TODO: Restore when vendored modules fixed
// use crate::vendored::glycan_gpu::{GlycanGpu, GlycanGpuConfig};
// use crate::vendored::cryptic_gpu::{CrypticGpu, CrypticGpuConfig};
// use crate::vendored::feature_merge::{FeatureMergeGpu, FeatureMergeConfig};
// use crate::vendored::fluxnet_gpu::{FluxNetGpu};
use crate::structure_types::{ParamyxoStructure, PackedResult};

/// High-level GPU pipeline orchestrator for NiV-Bench
pub struct ParallelGpuPipeline {
    pub device: Arc<CudaContext>,
    pub stream: Arc<CudaStream>, 
    mega_gpu: MegaFusedBatchGpu,
    glycan_gpu: GlycanGpu,
    cryptic_gpu: CrypticGpu,
    feature_merge: FeatureMergeGpu,
    pub fluxnet_gpu: FluxNetGpu,
}

impl ParallelGpuPipeline {
    pub fn new() -> Result<Self> {
        let device = CudaContext::new(0)?;
        let stream = device.new_stream()?;
        let ptx_dir = Path::new("kernels/ptx");

        log::info!("Loading PTX from: {:?}", ptx_dir);

        let mega_gpu = MegaFusedBatchGpu::new(device.clone(), MegaFusedConfig::default())?;
        let glycan_gpu = GlycanGpu::new(device.clone(), ptx_dir, GlycanGpuConfig::default())?;
        let cryptic_gpu = CrypticGpu::new(device.clone(), ptx_dir, CrypticGpuConfig::default())?;
        let feature_merge = FeatureMergeGpu::new(device.clone(), FeatureMergeConfig::default())?;
        let fluxnet_gpu = FluxNetGpu::new(device.clone(), ptx_dir)?;

        Ok(Self {
            device,
            stream,
            mega_gpu,
            glycan_gpu,
            cryptic_gpu,
            feature_merge,
            fluxnet_gpu,
        })
    }

    pub fn execute_batch(
        &mut self,
        batch: &mut PackedBatch,
        sequences: Option<&HashMap<String, String>>,
    ) -> Result<PackedResult> {
        let n_structures = batch.descriptors.len();
        let total_residues = batch.base.n_residues;

        if total_residues == 0 {
            return Ok(PackedResult::default());
        }

        // 1. Stage 0: Glycan Masking (Per Structure)
        // We use the first structure as a representative or iterate if needed
        // For simplicity, we process them sequentially on CPU for mask generation then upload
        let mut d_burial = self.stream.alloc_zeros::<f32>(total_residues)?;

        if let Some(seq_map) = sequences {
            for desc in &batch.descriptors {
                if let Some(seq) = seq_map.get(&desc.id) {
                    let seq_bytes = seq.as_bytes();
                    
                    // Coordinates for this structure
                    let start = desc.residue_start as usize;
                    let end = desc.residue_end as usize;
                    let n_res = end - start;
                    let struct_coords = &batch.base.ca_coords[start * 3 .. end * 3];

                    // Compute mask
                    let glycan_mask_vec = self.glycan_gpu.compute_mask(seq_bytes, struct_coords)?;
                    let glycan_mask_f32: Vec<f32> = glycan_mask_vec.iter().map(|&v| v as f32).collect();

                    // Upload mask to burial buffer slice
                    let mut d_burial_slice = d_burial.slice(start..end);
                    self.stream.memcpy_htod(&glycan_mask_f32, &mut d_burial_slice)?;
                    
                    log::info!("GPU Glycan Mask: {} residues shielded for {}", glycan_mask_vec.iter().filter(|&&v| v > 0).count(), desc.id);
                }
            }
        }

        // 2. Stage 1: Main Feature Extraction (136-dim)
        // Returns D_FEATURES_136 on GPU
        let d_features_136 = self.mega_gpu.detect_features_async(batch, Some(&d_burial))?;

        // 3. Stage 1.5: Cryptic Feature Extraction (4-dim)
        // For now, we use the CPU detect and upload results to stay compatible with existing CrypticGpu
        let mut cryptic_features_all = Vec::with_capacity(total_residues * 4);
        
        for desc in &batch.descriptors {
            let start = desc.residue_start as usize;
            let end = desc.residue_end as usize;
            let n_res = end - start;
            
            // Prepare coordinates as [[f32; 3]]
            let raw_coords = &batch.base.ca_coords[start * 3 .. end * 3];
            let coords_chunked: Vec<[f32; 3]> = raw_coords.chunks_exact(3)
                .map(|c| [c[0], c[1], c[2]])
                .collect();
                
            let res_indices: Vec<i32> = (0..n_res as i32).collect();

            let cryptic_result = self.cryptic_gpu.detect(
                &coords_chunked,
                &res_indices,
                &vec![0; n_res], // types
                &vec![0.0; n_res], // bfactors
                &vec![0.0; n_res], // hydro
                &vec![], // atoms
                &vec![],
                &vec![],
                &vec![],
            )?;

            // Interleave [Score, Mobility, Flexibility, Probe]
            for i in 0..n_res {
                cryptic_features_all.push(cryptic_result.residue_scores[i]);
                cryptic_features_all.push(cryptic_result.nma_mobility[i]);
                cryptic_features_all.push(cryptic_result.contact_order_flex[i]);
                cryptic_features_all.push(cryptic_result.probe_scores[i]);
            }
        }

        let mut d_features_4 = self.stream.alloc_zeros::<f32>(total_residues * 4)?;
        self.stream.memcpy_htod(&cryptic_features_all, &mut d_features_4)?;

        // 4. Stage 1.8: Feature Merge (136 + 4 -> 140)
        let d_features_merged = self.feature_merge.merge_features(&d_features_136, &d_features_4, total_residues)?;

        // 5. FluxNet-DQN Inference (Zero-Copy)
        let d_q_values = self.fluxnet_gpu.predict_batch(&d_features_merged, total_residues)?;

        // 6. Finalize Results
        let mut h_q_values = vec![0.0f32; total_residues * 4];
        self.stream.memcpy_dtoh(&d_q_values, &mut h_q_values)?;
        self.stream.synchronize()?;

        let mut results = PackedResult::default();
        for desc in &batch.descriptors {
            let start = desc.residue_start as usize;
            let end = desc.residue_end as usize;
            let q_slice = h_q_values[start * 4 .. end * 4].to_vec();

            results.structures.push(crate::structure_types::StructureResult {
                id: desc.id.clone(),
                q_values: q_slice,
                ..Default::default()
            });
        }

        Ok(results)
    }
}
