//! MegaFused Batch GPU (Vendored)

use anyhow::{Result, Context};
use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, DeviceSlice, DevicePtr, PushKernelArg, DeviceRepr};
use cudarc::driver::safe::ValidAsZeroBits;
use cudarc::driver::CudaContext;
use cudarc::nvrtc::Ptx;
use std::path::Path;
use std::sync::Arc;
use crate::vendored::mega_fused::{MegaFusedConfig, MegaFusedParams, GpuProvenanceData, GpuTelemetry};

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Default)]
pub struct BatchStructureDesc {
    pub atom_offset: i32,
    pub residue_offset: i32,
    pub n_atoms: i32,
    pub n_residues: i32,
}

#[derive(Debug, Clone)]
pub struct StructureInput {
    pub id: String,
    pub atoms: Vec<f32>,
    pub ca_indices: Vec<i32>,
    pub conservation: Vec<f32>,
    pub bfactor: Vec<f32>,
    pub burial: Vec<f32>,
    pub residue_types: Vec<i32>,
}

impl StructureInput {
    pub fn n_atoms(&self) -> usize { self.atoms.len() / 3 }
    pub fn n_residues(&self) -> usize { self.ca_indices.len() }
}

#[derive(Debug)]
pub struct PackedBatch {
    pub descriptors: Vec<BatchStructureDesc>,
    pub ids: Vec<String>,
    pub atoms_packed: Vec<f32>,
    pub ca_indices_packed: Vec<i32>,
    pub conservation_packed: Vec<f32>,
    pub bfactor_packed: Vec<f32>,
    pub burial_packed: Vec<f32>,
    pub residue_types_packed: Vec<i32>,
    pub total_atoms: usize,
    pub total_residues: usize,
    pub frequencies_packed: Vec<f32>,
    pub velocities_packed: Vec<f32>,
    pub p_neut_time_series_75pk_packed: Vec<f32>,
    pub current_immunity_levels_75_packed: Vec<f32>,
    pub pk_params_packed: Vec<f32>,
    pub epitope_escape_packed: Vec<f32>,
}

impl PackedBatch {
    pub fn n_structures(&self) -> usize { self.descriptors.len() }
}

#[derive(Debug, Clone)]
pub struct BatchStructureOutput {
    pub id: String,
    pub consensus_scores: Vec<f32>,
    pub confidence: Vec<i32>,
    pub signal_mask: Vec<i32>,
    pub pocket_assignment: Vec<i32>,
    pub centrality: Vec<f32>,
    pub combined_features: Vec<f32>,
    pub q_values: Vec<f32>, // FluxNet-DQN output
}

#[derive(Debug)]
pub struct BatchOutput {
    pub structures: Vec<BatchStructureOutput>,
    pub gpu_telemetry: Option<GpuProvenanceData>,
}

pub struct MegaFusedBatchGpu {
    device: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    batch_func: CudaFunction,
}

impl MegaFusedBatchGpu {
    pub fn new(device: Arc<CudaContext>, ptx_dir: &Path) -> Result<Self> {
        let stream = device.new_stream()?;
        
        let ptx_path = ptx_dir.join("mega_fused_batch.ptx");
        let ptx_src = std::fs::read_to_string(&ptx_path)
            .with_context(|| format!("Failed to read {}", ptx_path.display()))?;
            
        let ptx = Ptx::from_src(ptx_src);
        let module = device.load_module(ptx)?;
        let batch_func = module.load_function("mega_fused_batch_detection")
            .context("Failed to load kernel")?;

        Ok(Self {
            device,
            stream,
            batch_func,
        })
    }

    /// Run detection and output features directly to GPU memory (Zero-Copy)
    pub fn detect_features_async(
        &mut self, 
        batch: &PackedBatch, 
        optional_burial: Option<&CudaSlice<f32>>,
    ) -> Result<CudaSlice<f32>> {
        let n_structures = batch.n_structures();
        if n_structures == 0 { 
            return Ok(self.stream.alloc_zeros::<f32>(0)?); 
        }

        let total_atoms = batch.total_atoms;
        let total_residues = batch.total_residues;

        // Allocate output
        let mut d_feat_out = self.stream.alloc_zeros::<f32>(total_residues * 136)?;

        // Allocate and copy inputs
        let mut d_atoms = self.stream.alloc_zeros::<f32>(total_atoms * 3)?;
        self.stream.memcpy_htod(&batch.atoms_packed, &mut d_atoms)?;

        let mut d_ca = self.stream.alloc_zeros::<i32>(total_residues)?;
        self.stream.memcpy_htod(&batch.ca_indices_packed, &mut d_ca)?;

        let mut d_cons = self.stream.alloc_zeros::<f32>(total_residues)?;
        self.stream.memcpy_htod(&batch.conservation_packed, &mut d_cons)?;

        let mut d_bfactor = self.stream.alloc_zeros::<f32>(total_residues)?;
        self.stream.memcpy_htod(&batch.bfactor_packed, &mut d_bfactor)?;

        let mut d_res_types = self.stream.alloc_zeros::<i32>(total_residues)?;
        self.stream.memcpy_htod(&batch.residue_types_packed, &mut d_res_types)?;

        // Burial - Use provided GPU slice if available, otherwise upload from batch
        let d_burial = if let Some(buried) = optional_burial {
            buried.clone() // shallow copy of the handle
        } else {
            let mut b = self.stream.alloc_zeros::<f32>(total_residues)?;
            self.stream.memcpy_htod(&batch.burial_packed, &mut b)?;
            b
        };

        let desc_size = n_structures * std::mem::size_of::<BatchStructureDesc>();
        let mut d_desc = self.stream.alloc_zeros::<u8>(desc_size)?;
        let desc_bytes: &[u8] = unsafe {
             std::slice::from_raw_parts(batch.descriptors.as_ptr() as *const u8, desc_size)
        };
        self.stream.memcpy_htod(desc_bytes, &mut d_desc)?;

        // Temp Output buffers (required by kernel signature)
        let mut d_score = self.stream.alloc_zeros::<f32>(total_residues)?;
        let mut d_conf = self.stream.alloc_zeros::<i32>(total_residues)?;
        let mut d_mask = self.stream.alloc_zeros::<i32>(total_residues)?;
        let mut d_pocket = self.stream.alloc_zeros::<i32>(total_residues)?;
        let mut d_cent = self.stream.alloc_zeros::<f32>(total_residues)?;
        
        // Params
        let config = MegaFusedConfig::default();
        let params = MegaFusedParams::from_config(&config);
        let params_size = std::mem::size_of::<MegaFusedParams>();
        let mut d_params = self.stream.alloc_zeros::<u8>(params_size)?;
        let params_bytes: &[u8] = unsafe {
             std::slice::from_raw_parts(&params as *const _ as *const u8, params_size)
        };
        self.stream.memcpy_htod(params_bytes, &mut d_params)?;

        let launch_config = LaunchConfig {
            grid_dim: (n_structures as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        // Dummies
        let null_ptr: u64 = 0;
        let zero = 0i32;
        let fzero = 0.0f32;
        let d_dummy_f32 = self.stream.alloc_zeros::<f32>(1)?;

        unsafe {
            self.stream.launch_builder(&self.batch_func)
                .arg(&d_atoms).arg(&d_ca).arg(&d_cons).arg(&d_bfactor).arg(&d_burial).arg(&d_res_types).arg(&d_desc)
                .arg(&(n_structures as i32))
                .arg(&d_score).arg(&d_conf).arg(&d_mask).arg(&d_pocket).arg(&d_cent).arg(&mut d_feat_out)
                
                // Extra args
                .arg(&d_dummy_f32).arg(&d_dummy_f32) 
                .arg(&null_ptr).arg(&null_ptr).arg(&zero) 
                .arg(&600i32).arg(&0i32) 
                .arg(&null_ptr).arg(&null_ptr).arg(&null_ptr).arg(&0i32) 
                .arg(&d_dummy_f32).arg(&d_dummy_f32).arg(&d_dummy_f32) 
                .arg(&zero).arg(&1i32) 
                .arg(&fzero).arg(&fzero).arg(&fzero).arg(&fzero).arg(&fzero) 
                
                .arg(&d_params)
                .launch(launch_config)?;
        }
        
        Ok(d_feat_out)
    }
}
