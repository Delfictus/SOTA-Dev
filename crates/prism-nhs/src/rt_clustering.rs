//! RT-Core Accelerated Spatial Clustering
//!
//! Hardware-accelerated spatial clustering using NVIDIA RTX RT cores.
//! Replaces O(N²) CPU DBSCAN with O(N) GPU spatial queries.
//!
//! Pipeline:
//! 1. Build BVH from event positions (spheres)
//! 2. Launch RT kernel for neighbor finding
//! 3. GPU Union-Find for connected components
//! 4. Filter small clusters

use anyhow::{Context, Result};
use std::path::Path;
use std::sync::Arc;

#[cfg(feature = "gpu")]
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, CudaFunction, CudaModule, LaunchConfig, PushKernelArg};
#[cfg(feature = "gpu")]
use cudarc::nvrtc::Ptx;

#[cfg(feature = "gpu")]
use prism_optix::{
    AccelStructure, BvhBuildFlags, Module, ModuleCompileOptions, OptixContext,
    Pipeline, PipelineCompileOptions, PipelineLinkOptions, ProgramGroup,
    ShaderBindingTable, SBT_RECORD_HEADER_SIZE, aligned_sbt_record_size,
};
#[cfg(feature = "gpu")]
use optix_sys::CUstream;

/// RT clustering configuration
#[derive(Debug, Clone)]
pub struct RtClusteringConfig {
    /// Neighborhood radius (Å)
    pub epsilon: f32,
    /// Minimum points to form a core point
    pub min_points: u32,
    /// Minimum cluster size to keep
    pub min_cluster_size: u32,
    /// Rays per event for neighbor finding
    pub rays_per_event: u32,
}

impl Default for RtClusteringConfig {
    fn default() -> Self {
        Self {
            epsilon: 5.0,
            min_points: 3,
            min_cluster_size: 100,
            rays_per_event: 64,
        }
    }
}

/// Launch parameters for RT clustering kernel
/// Must match RtClusteringParams in rt_clustering.cu exactly (88 bytes)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RtClusteringParams {
    // BVH for event positions
    pub traversable: u64,           // OptixTraversableHandle (8 bytes)

    // Input: Event positions
    pub event_positions: u64,       // float3* device pointer (8 bytes)
    pub num_events: u32,            // (4 bytes)

    // Clustering parameters (no padding needed - naturally aligned)
    pub epsilon: f32,               // (4 bytes)
    pub min_points: u32,            // (4 bytes)
    pub rays_per_event: u32,        // (4 bytes)

    // Output: Neighbor counts and indices (all pointers are 8-byte aligned after 16 bytes above)
    pub neighbor_counts: u64,       // unsigned int* (8 bytes)
    pub neighbor_offsets: u64,      // unsigned int* (8 bytes)
    pub neighbor_indices: u64,      // unsigned int* (8 bytes)

    // Output: Cluster assignments
    pub cluster_ids: u64,           // int* (8 bytes)
    pub parent: u64,                // int* (8 bytes)

    // Statistics
    pub total_neighbors: u64,       // unsigned int* (8 bytes)
    pub num_clusters: u64,          // unsigned int* (8 bytes)
    // Total: 8+8+4+4+4+4+8+8+8+8+8+8+8 = 88 bytes
}

/// Result of RT clustering
#[derive(Debug, Clone)]
pub struct RtClusteringResult {
    /// Cluster ID for each event (-1 = noise)
    pub cluster_ids: Vec<i32>,
    /// Number of clusters found
    pub num_clusters: usize,
    /// Total neighbor pairs found
    pub total_neighbors: usize,
    /// GPU time in milliseconds
    pub gpu_time_ms: f64,
}

/// RT-Core accelerated clustering engine
#[cfg(feature = "gpu")]
pub struct RtClusteringEngine {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    optix_ctx: OptixContext,
    config: RtClusteringConfig,

    // OptiX pipeline components
    module: Option<Module>,
    raygen_pg: Option<ProgramGroup>,
    miss_pg: Option<ProgramGroup>,
    hitgroup_pg: Option<ProgramGroup>,
    pipeline: Option<Pipeline>,

    // SBT records (device memory)
    d_raygen_record: Option<CudaSlice<u8>>,
    d_miss_record: Option<CudaSlice<u8>>,
    d_hitgroup_record: Option<CudaSlice<u8>>,

    // Phase 2: Build neighbor list (different raygen)
    raygen_build_pg: Option<ProgramGroup>,
    miss_build_pg: Option<ProgramGroup>,
    hitgroup_build_pg: Option<ProgramGroup>,
    pipeline_build: Option<Pipeline>,
    d_raygen_build_record: Option<CudaSlice<u8>>,
    d_miss_build_record: Option<CudaSlice<u8>>,
    d_hitgroup_build_record: Option<CudaSlice<u8>>,

    // CUDA module and kernels for union-find clustering
    cuda_module: Option<Arc<CudaModule>>,
    fn_compute_offsets: Option<CudaFunction>,
    fn_init_union_find: Option<CudaFunction>,
    fn_union_neighbors: Option<CudaFunction>,
    fn_flatten_clusters: Option<CudaFunction>,
    fn_propagate_ids: Option<CudaFunction>,
    fn_count_sizes: Option<CudaFunction>,
    fn_filter_small: Option<CudaFunction>,
}

#[cfg(feature = "gpu")]
impl RtClusteringEngine {
    /// Create a new RT clustering engine
    pub fn new(context: Arc<CudaContext>, config: RtClusteringConfig) -> Result<Self> {
        // Initialize OptiX
        OptixContext::init()
            .map_err(|e| anyhow::anyhow!("OptiX init failed: {}", e))?;

        let optix_ctx = OptixContext::new(context.cu_ctx(), false)
            .map_err(|e| anyhow::anyhow!("OptiX context failed: {}", e))?;

        let stream = context.default_stream();

        log::info!("RT clustering engine created");

        Ok(Self {
            context,
            stream,
            optix_ctx,
            config,
            module: None,
            raygen_pg: None,
            miss_pg: None,
            hitgroup_pg: None,
            pipeline: None,
            d_raygen_record: None,
            d_miss_record: None,
            d_hitgroup_record: None,
            raygen_build_pg: None,
            miss_build_pg: None,
            hitgroup_build_pg: None,
            pipeline_build: None,
            d_raygen_build_record: None,
            d_miss_build_record: None,
            d_hitgroup_build_record: None,
            cuda_module: None,
            fn_compute_offsets: None,
            fn_init_union_find: None,
            fn_union_neighbors: None,
            fn_flatten_clusters: None,
            fn_propagate_ids: None,
            fn_count_sizes: None,
            fn_filter_small: None,
        })
    }

    /// Load the RT clustering pipeline from OptiX IR
    pub fn load_pipeline(&mut self, optixir_path: impl AsRef<Path>) -> Result<()> {
        let path = optixir_path.as_ref();
        log::info!("Loading RT clustering pipeline from: {}", path.display());

        // Module compile options
        let module_options = ModuleCompileOptions::default();

        // Pipeline compile options
        let mut pipeline_options = PipelineCompileOptions::default();
        pipeline_options.num_payload_values = 2;  // hit_event_id, hit_count
        pipeline_options.num_attribute_values = 2;

        // Debug: log the size of our params struct
        let params_size = std::mem::size_of::<RtClusteringParams>();
        log::info!("RT clustering params struct size: {} bytes", params_size);

        // Load module from OptiX IR
        let module = Module::from_optix_ir(
            &self.optix_ctx,
            path,
            &module_options,
            &pipeline_options,
        ).context("Failed to load OptiX IR module")?;

        // Create program groups
        let raygen_pg = ProgramGroup::create_raygen(
            &self.optix_ctx,
            &module,
            "__raygen__count_neighbors",
        ).context("Failed to create raygen program group")?;

        let miss_pg = ProgramGroup::create_miss(
            &self.optix_ctx,
            &module,
            "__miss__count_neighbors",
        ).context("Failed to create miss program group")?;

        let hitgroup_pg = ProgramGroup::create_hitgroup(
            &self.optix_ctx,
            Some(&module),
            Some("__closesthit__count_neighbors"),
            None, None,  // No any-hit
            None, None,  // No intersection (using built-in spheres)
        ).context("Failed to create hitgroup program group")?;

        // Create pipeline
        let link_options = PipelineLinkOptions::default();
        let pipeline = Pipeline::create(
            &self.optix_ctx,
            &pipeline_options,
            &link_options,
            &[&raygen_pg, &miss_pg, &hitgroup_pg],
        ).context("Failed to create pipeline")?;

        // Set stack sizes (last param must be 1 for SINGLE_GAS traversable graph)
        pipeline.set_stack_size(0, 0, 1, 1)
            .context("Failed to set pipeline stack size")?;

        // Create SBT records
        self.setup_sbt(&raygen_pg, &miss_pg, &hitgroup_pg)?;

        // Store components
        self.module = Some(module);
        self.raygen_pg = Some(raygen_pg);
        self.miss_pg = Some(miss_pg);
        self.hitgroup_pg = Some(hitgroup_pg);
        self.pipeline = Some(pipeline);

        // Create Phase 2 pipeline for building neighbor list
        // (reuses the same module but different entry points)
        let module_ref = self.module.as_ref().unwrap();

        let raygen_build_pg = ProgramGroup::create_raygen(
            &self.optix_ctx,
            module_ref,
            "__raygen__build_neighbors",
        ).context("Failed to create build raygen program group")?;

        let miss_build_pg = ProgramGroup::create_miss(
            &self.optix_ctx,
            module_ref,
            "__miss__build_neighbors",
        ).context("Failed to create build miss program group")?;

        let hitgroup_build_pg = ProgramGroup::create_hitgroup(
            &self.optix_ctx,
            Some(module_ref),
            Some("__closesthit__build_neighbors"),
            None, None,
            None, None,
        ).context("Failed to create build hitgroup program group")?;

        let pipeline_build = Pipeline::create(
            &self.optix_ctx,
            &pipeline_options,
            &link_options,
            &[&raygen_build_pg, &miss_build_pg, &hitgroup_build_pg],
        ).context("Failed to create build pipeline")?;

        pipeline_build.set_stack_size(0, 0, 1, 1)
            .context("Failed to set build pipeline stack size")?;

        // Setup SBT for build phase
        self.setup_sbt_build(&raygen_build_pg, &miss_build_pg, &hitgroup_build_pg)?;

        self.raygen_build_pg = Some(raygen_build_pg);
        self.miss_build_pg = Some(miss_build_pg);
        self.hitgroup_build_pg = Some(hitgroup_build_pg);
        self.pipeline_build = Some(pipeline_build);

        log::info!("Phase 2 (build neighbors) pipeline created");

        // Load PTX for regular CUDA kernels (separate file from OptiX shaders)
        let ptx_path = path.with_file_name("rt_clustering_cuda.ptx");
        if ptx_path.exists() {
            log::info!("Loading CUDA kernels from: {}", ptx_path.display());
            let ptx = Ptx::from_file(&ptx_path);

            let cuda_module = self.context
                .load_module(ptx)
                .context("Failed to load rt_clustering CUDA module")?;

            self.fn_compute_offsets = Some(cuda_module.load_function("compute_neighbor_offsets")
                .context("Failed to load compute_neighbor_offsets")?);
            self.fn_init_union_find = Some(cuda_module.load_function("init_union_find")
                .context("Failed to load init_union_find")?);
            self.fn_union_neighbors = Some(cuda_module.load_function("union_neighbors")
                .context("Failed to load union_neighbors")?);
            self.fn_flatten_clusters = Some(cuda_module.load_function("flatten_clusters")
                .context("Failed to load flatten_clusters")?);
            self.fn_propagate_ids = Some(cuda_module.load_function("propagate_cluster_ids")
                .context("Failed to load propagate_cluster_ids")?);
            self.fn_count_sizes = Some(cuda_module.load_function("count_cluster_sizes")
                .context("Failed to load count_cluster_sizes")?);
            self.fn_filter_small = Some(cuda_module.load_function("filter_small_clusters")
                .context("Failed to load filter_small_clusters")?);
            self.cuda_module = Some(cuda_module);
            log::info!("CUDA union-find kernels loaded");
        } else {
            log::warn!("PTX file not found: {} - clustering will only count neighbors", ptx_path.display());
        }

        log::info!("RT clustering pipeline loaded successfully");
        Ok(())
    }

    /// Setup Shader Binding Table
    fn setup_sbt(
        &mut self,
        raygen_pg: &ProgramGroup,
        miss_pg: &ProgramGroup,
        hitgroup_pg: &ProgramGroup,
    ) -> Result<()> {
        let record_size = aligned_sbt_record_size(0); // Header only, no data

        // Raygen record
        let mut raygen_record = vec![0u8; record_size];
        raygen_pg.pack_header(&mut raygen_record)
            .map_err(|e| anyhow::anyhow!("Failed to pack raygen header: {}", e))?;
        self.d_raygen_record = Some(self.stream.clone_htod(&raygen_record)?);

        // Miss record
        let mut miss_record = vec![0u8; record_size];
        miss_pg.pack_header(&mut miss_record)
            .map_err(|e| anyhow::anyhow!("Failed to pack miss header: {}", e))?;
        self.d_miss_record = Some(self.stream.clone_htod(&miss_record)?);

        // Hitgroup record
        let mut hitgroup_record = vec![0u8; record_size];
        hitgroup_pg.pack_header(&mut hitgroup_record)
            .map_err(|e| anyhow::anyhow!("Failed to pack hitgroup header: {}", e))?;
        self.d_hitgroup_record = Some(self.stream.clone_htod(&hitgroup_record)?);

        Ok(())
    }

    /// Setup Shader Binding Table for build phase
    fn setup_sbt_build(
        &mut self,
        raygen_pg: &ProgramGroup,
        miss_pg: &ProgramGroup,
        hitgroup_pg: &ProgramGroup,
    ) -> Result<()> {
        let record_size = aligned_sbt_record_size(0);

        let mut raygen_record = vec![0u8; record_size];
        raygen_pg.pack_header(&mut raygen_record)
            .map_err(|e| anyhow::anyhow!("Failed to pack build raygen header: {}", e))?;
        self.d_raygen_build_record = Some(self.stream.clone_htod(&raygen_record)?);

        let mut miss_record = vec![0u8; record_size];
        miss_pg.pack_header(&mut miss_record)
            .map_err(|e| anyhow::anyhow!("Failed to pack build miss header: {}", e))?;
        self.d_miss_build_record = Some(self.stream.clone_htod(&miss_record)?);

        let mut hitgroup_record = vec![0u8; record_size];
        hitgroup_pg.pack_header(&mut hitgroup_record)
            .map_err(|e| anyhow::anyhow!("Failed to pack build hitgroup header: {}", e))?;
        self.d_hitgroup_build_record = Some(self.stream.clone_htod(&hitgroup_record)?);

        Ok(())
    }

    /// Cluster positions using RT cores
    /// positions: [x0, y0, z0, x1, y1, z1, ...] flattened float3 array
    pub fn cluster(&self, positions: &[f32]) -> Result<RtClusteringResult> {
        let num_events = positions.len() / 3;
        if num_events == 0 {
            return Ok(RtClusteringResult {
                cluster_ids: vec![],
                num_clusters: 0,
                total_neighbors: 0,
                gpu_time_ms: 0.0,
            });
        }

        let pipeline = self.pipeline.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Pipeline not loaded. Call load_pipeline() first."))?;

        let start = std::time::Instant::now();

        // Upload positions to GPU
        let d_positions: CudaSlice<f32> = self.stream.clone_htod(&positions.to_vec())?;

        // Create radii buffer (all epsilon/2 for neighbor overlap detection)
        let radii: Vec<f32> = vec![self.config.epsilon / 2.0; num_events];
        let d_radii: CudaSlice<f32> = self.stream.clone_htod(&radii)?;

        // Get device pointers for BVH build
        let (positions_ptr, _guard1) = d_positions.device_ptr(&self.stream);
        let (radii_ptr, _guard2) = d_radii.device_ptr(&self.stream);

        // Build BVH from positions
        let bvh = AccelStructure::build_spheres(
            &self.optix_ctx,
            positions_ptr,
            radii_ptr,
            num_events,
            BvhBuildFlags::dynamic(),
        ).map_err(|e| anyhow::anyhow!("BVH build failed: {}", e))?;

        // Allocate output buffers
        let zeros_u32: Vec<u32> = vec![0; num_events];
        let zeros_u32_plus1: Vec<u32> = vec![0; num_events + 1];
        let initial_parent: Vec<i32> = (0..num_events as i32).collect();
        let noise_ids: Vec<i32> = vec![-1; num_events];

        let mut d_neighbor_counts: CudaSlice<u32> = self.stream.clone_htod(&zeros_u32.clone())?;
        let d_neighbor_offsets: CudaSlice<u32> = self.stream.clone_htod(&zeros_u32_plus1)?;
        let d_cluster_ids: CudaSlice<i32> = self.stream.clone_htod(&noise_ids)?;
        let d_parent: CudaSlice<i32> = self.stream.clone_htod(&initial_parent)?;
        let d_total_neighbors: CudaSlice<u32> = self.stream.clone_htod(&vec![0u32])?;
        let d_num_clusters: CudaSlice<u32> = self.stream.clone_htod(&vec![0u32])?;

        // Pre-allocate neighbor indices (estimate max neighbors)
        let max_neighbors = num_events * 32; // Estimate
        let d_neighbor_indices: CudaSlice<u32> = self.stream.clone_htod(&vec![0u32; max_neighbors])?;

        // Get device pointers for params
        // Note: We extract the raw pointer values and sync to ensure they're valid
        // The SyncOnDrop guards are scoped to drop immediately after extraction
        let positions_ptr2 = { let (p, _) = d_positions.device_ptr(&self.stream); p };
        let neighbor_counts_ptr = { let (p, _) = d_neighbor_counts.device_ptr(&self.stream); p };
        let neighbor_offsets_ptr = { let (p, _) = d_neighbor_offsets.device_ptr(&self.stream); p };
        let neighbor_indices_ptr = { let (p, _) = d_neighbor_indices.device_ptr(&self.stream); p };
        let cluster_ids_ptr = { let (p, _) = d_cluster_ids.device_ptr(&self.stream); p };
        let parent_ptr = { let (p, _) = d_parent.device_ptr(&self.stream); p };
        let total_neighbors_ptr = { let (p, _) = d_total_neighbors.device_ptr(&self.stream); p };
        let num_clusters_ptr = { let (p, _) = d_num_clusters.device_ptr(&self.stream); p };

        // Setup launch parameters (88 bytes, must match rt_clustering.cu exactly)
        let params = RtClusteringParams {
            traversable: bvh.handle(),
            event_positions: positions_ptr2,
            num_events: num_events as u32,
            epsilon: self.config.epsilon,
            min_points: self.config.min_points,
            rays_per_event: self.config.rays_per_event,
            neighbor_counts: neighbor_counts_ptr,
            neighbor_offsets: neighbor_offsets_ptr,
            neighbor_indices: neighbor_indices_ptr,
            cluster_ids: cluster_ids_ptr,
            parent: parent_ptr,
            total_neighbors: total_neighbors_ptr,
            num_clusters: num_clusters_ptr,
        };

        // Upload params to device (as raw bytes since RtClusteringParams doesn't impl DeviceRepr)
        let params_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                &params as *const RtClusteringParams as *const u8,
                std::mem::size_of::<RtClusteringParams>(),
            )
        };
        let d_params: CudaSlice<u8> = self.stream.clone_htod(params_bytes)?;
        let (params_ptr, _gp) = d_params.device_ptr(&self.stream);

        // Build SBT
        let record_size = aligned_sbt_record_size(0) as u32;
        let (raygen_ptr, _gr) = self.d_raygen_record.as_ref().unwrap().device_ptr(&self.stream);
        let (miss_ptr, _gm) = self.d_miss_record.as_ref().unwrap().device_ptr(&self.stream);
        let (hitgroup_ptr, _gh) = self.d_hitgroup_record.as_ref().unwrap().device_ptr(&self.stream);

        let sbt = ShaderBindingTable {
            raygen_record: raygen_ptr,
            exception_record: 0,
            miss_record_base: miss_ptr,
            miss_record_stride: record_size,
            miss_record_count: 1,
            hitgroup_record_base: hitgroup_ptr,
            hitgroup_record_stride: record_size,
            hitgroup_record_count: 1,
            callable_record_base: 0,
            callable_record_stride: 0,
            callable_record_count: 0,
        };

        // Launch pipeline
        // Cast between cudarc and optix-sys CUstream types (same underlying type, different FFI bindings)
        let cu_stream = self.stream.cu_stream() as CUstream;
        pipeline.launch(
            cu_stream,
            params_ptr,
            std::mem::size_of::<RtClusteringParams>(),
            &sbt,
            num_events as u32,        // width = num_events
            self.config.rays_per_event, // height = rays_per_event
            1,                        // depth = 1
        ).map_err(|e| anyhow::anyhow!("Pipeline launch failed: {}", e))?;

        // Synchronize after Phase 1
        self.stream.synchronize()?;

        // DEBUG: Check neighbor counts from Phase 1
        let mut neighbor_counts_host = vec![0u32; num_events];
        self.stream.memcpy_dtoh(&d_neighbor_counts, &mut neighbor_counts_host)?;
        let total_raw_neighbors: u32 = neighbor_counts_host.iter().sum();
        log::info!(
            "Phase 1 complete: {} raw neighbor hits from {} events",
            total_raw_neighbors, num_events
        );

        // If CUDA kernels are loaded, run the full clustering pipeline
        let (final_cluster_ids, final_total_neighbors, final_num_clusters) = if self.cuda_module.is_some() {
            // Phase 2: Compute CSR offsets (prefix sum)
            // This kernel runs on a single thread to do sequential prefix sum
            let fn_offsets = self.fn_compute_offsets.as_ref().unwrap();
            let num_events_u32 = num_events as u32;
            unsafe {
                self.stream.launch_builder(fn_offsets)
                    .arg(&d_neighbor_counts)
                    .arg(&d_neighbor_offsets)
                    .arg(&num_events_u32)
                    .arg(&d_total_neighbors)
                    .launch(LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .context("Failed to launch compute_neighbor_offsets")?;
            }
            self.stream.synchronize()?;

            // Get total neighbors for sanity check
            let mut total_neighbors_host = vec![0u32; 1];
            self.stream.memcpy_dtoh(&d_total_neighbors, &mut total_neighbors_host)?;
            let total_neighbors_val = total_neighbors_host[0] as usize;
            log::info!("Phase 2 complete: {} total neighbor pairs", total_neighbors_val);

            // Phase 3: Build neighbor list (second OptiX launch)
            // CRITICAL: Reset neighbor_counts to 0 before Phase 3
            // The closesthit__build_neighbors kernel uses atomicAdd on neighbor_counts as a write cursor
            // If we don't reset it, the counts from Phase 1 cause buffer overruns
            let zeros_for_reset = vec![0u32; num_events];
            self.stream.memcpy_htod(&zeros_for_reset, &mut d_neighbor_counts)?;

            let pipeline_build = self.pipeline_build.as_ref().unwrap();
            let (raygen_build_ptr, _grb) = self.d_raygen_build_record.as_ref().unwrap().device_ptr(&self.stream);
            let (miss_build_ptr, _gmb) = self.d_miss_build_record.as_ref().unwrap().device_ptr(&self.stream);
            let (hitgroup_build_ptr, _ghb) = self.d_hitgroup_build_record.as_ref().unwrap().device_ptr(&self.stream);

            let sbt_build = ShaderBindingTable {
                raygen_record: raygen_build_ptr,
                exception_record: 0,
                miss_record_base: miss_build_ptr,
                miss_record_stride: record_size,
                miss_record_count: 1,
                hitgroup_record_base: hitgroup_build_ptr,
                hitgroup_record_stride: record_size,
                hitgroup_record_count: 1,
                callable_record_base: 0,
                callable_record_stride: 0,
                callable_record_count: 0,
            };

            pipeline_build.launch(
                cu_stream,
                params_ptr,
                std::mem::size_of::<RtClusteringParams>(),
                &sbt_build,
                num_events as u32,
                self.config.rays_per_event,
                1,
            ).map_err(|e| anyhow::anyhow!("Build pipeline launch failed: {}", e))?;
            self.stream.synchronize()?;
            log::info!("Phase 3 complete: neighbor list built");

            // Phase 4: Initialize union-find
            let fn_init = self.fn_init_union_find.as_ref().unwrap();
            let blocks = ((num_events + 255) / 256) as u32;
            unsafe {
                self.stream.launch_builder(fn_init)
                    .arg(&d_parent)
                    .arg(&num_events_u32)
                    .launch(LaunchConfig {
                        grid_dim: (blocks, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .context("Failed to launch init_union_find")?;
            }

            // Phase 5: Union neighbors
            // Kernel signature: (parent, neighbor_counts, neighbor_offsets, neighbor_indices, num_events)
            let fn_union = self.fn_union_neighbors.as_ref().unwrap();
            unsafe {
                self.stream.launch_builder(fn_union)
                    .arg(&d_parent)
                    .arg(&d_neighbor_counts)
                    .arg(&d_neighbor_offsets)
                    .arg(&d_neighbor_indices)
                    .arg(&num_events_u32)
                    .launch(LaunchConfig {
                        grid_dim: (blocks, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .context("Failed to launch union_neighbors")?;
            }

            // Phase 6: Flatten clusters (run multiple times for convergence)
            let fn_flatten = self.fn_flatten_clusters.as_ref().unwrap();
            for _ in 0..3 {
                unsafe {
                    self.stream.launch_builder(fn_flatten)
                        .arg(&d_parent)
                        .arg(&num_events_u32)
                        .launch(LaunchConfig {
                            grid_dim: (blocks, 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        })
                        .context("Failed to launch flatten_clusters")?;
                }
            }

            // Phase 7: Propagate cluster IDs
            // Kernel signature: (parent, cluster_ids, num_events, num_clusters)
            let fn_propagate = self.fn_propagate_ids.as_ref().unwrap();
            unsafe {
                self.stream.launch_builder(fn_propagate)
                    .arg(&d_parent)
                    .arg(&d_cluster_ids)
                    .arg(&num_events_u32)
                    .arg(&d_num_clusters)
                    .launch(LaunchConfig {
                        grid_dim: (blocks, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .context("Failed to launch propagate_cluster_ids")?;
            }
            self.stream.synchronize()?;
            log::info!("Phase 7 complete: cluster IDs assigned");

            // Download results
            let mut cluster_ids = vec![-1i32; num_events];
            self.stream.memcpy_dtoh(&d_cluster_ids, &mut cluster_ids)?;

            // Count unique clusters
            let mut unique_clusters: std::collections::HashSet<i32> = std::collections::HashSet::new();
            for &cid in &cluster_ids {
                if cid >= 0 {
                    unique_clusters.insert(cid);
                }
            }

            (cluster_ids, total_neighbors_val, unique_clusters.len())
        } else {
            // Fallback: just return raw counts without clustering
            log::warn!("CUDA kernels not loaded - returning neighbor counts only");
            let cluster_ids = vec![-1i32; num_events];
            (cluster_ids, total_raw_neighbors as usize, 0)
        };

        let gpu_time = start.elapsed().as_secs_f64() * 1000.0;

        log::info!(
            "RT clustering: {} events, {} neighbors, {} clusters, {:.2}ms",
            num_events, final_total_neighbors, final_num_clusters, gpu_time
        );

        Ok(RtClusteringResult {
            cluster_ids: final_cluster_ids,
            num_clusters: final_num_clusters,
            total_neighbors: final_total_neighbors,
            gpu_time_ms: gpu_time,
        })
    }
}

/// Find the rt_clustering.optixir file
pub fn find_optixir_path() -> Option<std::path::PathBuf> {
    // Try relative paths from common locations
    let candidates = [
        "crates/prism-gpu/src/kernels/rt_clustering.optixir",
        "../prism-gpu/src/kernels/rt_clustering.optixir",
        "../../prism-gpu/src/kernels/rt_clustering.optixir",
    ];

    for path in &candidates {
        let p = std::path::Path::new(path);
        if p.exists() {
            return Some(p.to_path_buf());
        }
    }

    // Try from CARGO_MANIFEST_DIR
    if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
        let p = std::path::Path::new(&manifest_dir)
            .parent()?
            .join("prism-gpu/src/kernels/rt_clustering.optixir");
        if p.exists() {
            return Some(p);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = RtClusteringConfig::default();
        assert_eq!(config.epsilon, 5.0);
        assert_eq!(config.min_points, 3);
        assert_eq!(config.rays_per_event, 64);
    }

    #[test]
    fn test_params_size() {
        // Verify struct is correctly sized for GPU (must match rt_clustering.cu exactly)
        // Layout: 8+8+4+4+4+4+8+8+8+8+8+8+8 = 88 bytes
        assert_eq!(std::mem::size_of::<RtClusteringParams>(), 88);
    }
}
