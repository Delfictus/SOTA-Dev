    /// Build immunity cache using PATH A: Epitope-based P_neut (TARGET: 85-90% accuracy)
    ///
    /// Key differences from PATH B:
    /// - Uses weighted epitope distance instead of PK pharmacokinetics
    /// - Precomputes P_neut matrix (variant × variant)
    /// - Simpler model: 12 parameters (11 weights + sigma) vs 75 PK combinations
    /// - Direct calibration to VASIL reference P_neut
    pub fn build_for_landscape_gpu_path_a(
        landscape: &ImmunityLandscape,
        dms_data: &DmsEscapeData,
        _pk: &PkParams,
        context: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        eval_start: NaiveDate,
        eval_end: NaiveDate,
        lineage_mutations: &HashMap<String, Vec<String>>,
        epitope_weights: &[f32; 11],  // NEW: Calibrated epitope weights
        sigma: f32,                    // NEW: Gaussian bandwidth
    ) -> Result<Self> {
        use cudarc::driver::LaunchConfig;
        use anyhow::anyhow;
        
        const N_EPITOPES: usize = 11;
        
        eprintln!("[ImmunityCache GPU PATH A] Epitope-based P_neut approach");
        eprintln!("  Weights: [{:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}]",
                  epitope_weights[0], epitope_weights[1], epitope_weights[2], epitope_weights[3],
                  epitope_weights[4], epitope_weights[5], epitope_weights[6], epitope_weights[7],
                  epitope_weights[8], epitope_weights[9], epitope_weights[10]);
        eprintln!("  Sigma: {:.3}", sigma);
        
        let start_time = std::time::Instant::now();
        
        // Filter significant variants (same as PATH B)
        let significant_indices: Vec<usize> = landscape.lineages.iter()
            .enumerate()
            .filter(|(idx, _)| {
                landscape.variant_frequencies.iter()
                    .filter_map(|day_freqs| day_freqs.get(*idx))
                    .any(|&f| f >= 0.01)
            })
            .map(|(idx, _)| idx)
            .collect();
        
        let n_variants = significant_indices.len();
        let n_eval_days = (eval_end - eval_start).num_days() as usize;
        
        let data_start = landscape.start_date;
        let eval_start_offset = (eval_start - data_start).num_days().max(0) as usize;
        let max_history_days = landscape.daily_incidence.len();
        
        eprintln!("  {} significant variants (of {} total)", n_variants, landscape.lineages.len());
        eprintln!("  {} eval days", n_eval_days);
        eprintln!("  {} days history, eval offset {}", max_history_days, eval_start_offset);
        
        // ═══════════════════════════════════════════════════════════════════
        // STEP 1: Extract epitope escape (11 epitopes) - SAME AS PATH B
        // ═══════════════════════════════════════════════════════════════════
        let mut epitope_escape = vec![0.0f32; n_variants * N_EPITOPES];
        for (new_idx, &orig_idx) in significant_indices.iter().enumerate() {
            let lineage = &landscape.lineages[orig_idx];
            for e in 0..10 {
                epitope_escape[new_idx * N_EPITOPES + e] =
                    dms_data.get_epitope_escape(lineage, e).unwrap_or(0.0);
            }
            // Epitope 10 = NTD
            let ntd = dms_data.get_ntd_escape(lineage).unwrap_or(0.4) as f32;
            epitope_escape[new_idx * N_EPITOPES + 10] = ntd;
        }
        
        // Upload epitope escape to GPU
        let mut d_epitope_escape = stream.alloc_zeros(n_variants * N_EPITOPES)
            .map_err(|e| anyhow!("Alloc epitope_escape: {}", e))?;
        stream.memcpy_htod(&epitope_escape, &mut d_epitope_escape)
            .map_err(|e| anyhow!("Upload epitope_escape: {}", e))?;
        
        // Upload epitope weights to GPU
        let mut d_epitope_weights = stream.alloc_zeros(N_EPITOPES)
            .map_err(|e| anyhow!("Alloc weights: {}", e))?;
        stream.memcpy_htod(epitope_weights, &mut d_epitope_weights)
            .map_err(|e| anyhow!("Upload weights: {}", e))?;
        
        // ═══════════════════════════════════════════════════════════════════
        // STEP 2: Compute P_neut matrix [n_variants × n_variants] on GPU
        // ═══════════════════════════════════════════════════════════════════
        eprintln!("[PATH A] Computing epitope-based P_neut matrix ({} × {})...", n_variants, n_variants);
        
        // Load PTX module
        let ptx_path = "target/ptx/epitope_p_neut.ptx";
        let ptx = std::fs::read_to_string(ptx_path)
            .map_err(|e| anyhow!("Failed to read {}: {}", ptx_path, e))?;
        
        context.load_ptx(ptx.into(), "epitope_p_neut", &["compute_epitope_p_neut"])
            .map_err(|e| anyhow!("Load PTX: {}", e))?;
        
        let compute_p_neut_func = context.get_func("epitope_p_neut", "compute_epitope_p_neut")
            .ok_or_else(|| anyhow!("Function compute_epitope_p_neut not found"))?;
        
        // Allocate P_neut matrix
        let mut d_p_neut_matrix = stream.alloc_zeros(n_variants * n_variants)
            .map_err(|e| anyhow!("Alloc P_neut matrix: {}", e))?;
        
        // Launch kernel: grid (n_variants, n_variants, 1), block (1, 1, 1)
        let cfg_p_neut = LaunchConfig {
            grid_dim: (n_variants as u32, n_variants as u32, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        
        let n_variants_i32 = n_variants as i32;
        unsafe {
            let mut builder = stream.launch_builder(&compute_p_neut_func);
            builder.arg(&d_epitope_escape);
            builder.arg(&d_p_neut_matrix);
            builder.arg(&d_epitope_weights);
            builder.arg(&sigma);
            builder.arg(&n_variants_i32);
            builder.launch(cfg_p_neut)
                .map_err(|e| anyhow!("Launch compute_epitope_p_neut: {}", e))?;
        }
        
        stream.synchronize().map_err(|e| anyhow!("Sync P_neut: {}", e))?;
        eprintln!("[PATH A] ✓ P_neut matrix computed");
        
        // Download P_neut matrix for inspection (optional - can skip for production)
        let p_neut_matrix: Vec<f32> = stream.clone_dtoh(&d_p_neut_matrix)
            .map_err(|e| anyhow!("Download P_neut: {}", e))?;
        
        // Diagnostic: Check P_neut values
        if n_variants > 0 {
            let p_self = p_neut_matrix[0];  // P_neut(0,0) - should be ~1.0
            let p_other = if n_variants > 1 { p_neut_matrix[1] } else { 0.0 };  // P_neut(0,1)
            eprintln!("[PATH A] P_neut diagnostics: self={:.4}, other={:.4}", p_self, p_other);
        }
        
        // ═══════════════════════════════════════════════════════════════════
        // STEP 3: Compute immunity using P_neut matrix (simplified vs PATH B)
        // ═══════════════════════════════════════════════════════════════════
        eprintln!("[PATH A] Computing immunity from P_neut matrix...");
        
        // Build frequency matrix [n_variants × max_history_days]
        let mut frequencies = vec![0.0f32; n_variants * max_history_days];
        for (new_idx, &orig_idx) in significant_indices.iter().enumerate() {
            for day_idx in 0..max_history_days {
                if let Some(day_freqs) = landscape.variant_frequencies.get(day_idx) {
                    if let Some(&freq) = day_freqs.get(orig_idx) {
                        frequencies[new_idx * max_history_days + day_idx] = freq;
                    }
                }
            }
        }
        
        let mut d_frequencies = stream.alloc_zeros(n_variants * max_history_days)
            .map_err(|e| anyhow!("Alloc frequencies: {}", e))?;
        stream.memcpy_htod(&frequencies, &mut d_frequencies)
            .map_err(|e| anyhow!("Upload frequencies: {}", e))?;
        
        // Upload incidence
        let mut d_incidence = stream.alloc_zeros(max_history_days)
            .map_err(|e| anyhow!("Alloc incidence: {}", e))?;
        stream.memcpy_htod(&landscape.daily_incidence, &mut d_incidence)
            .map_err(|e| anyhow!("Upload incidence: {}", e))?;
        
        // Allocate immunity output [n_variants × n_eval_days]
        let mut d_immunity = stream.alloc_zeros(n_variants * n_eval_days)
            .map_err(|e| anyhow!("Alloc immunity: {}", e))?;
        
        // For PATH A, we compute a SINGLE immunity value per (variant, day)
        // This is simpler than PATH B's 75 PK combinations
        // We'll compute it on CPU for now (can GPU-accelerate later if needed)
        
        let mut immunity_matrix = vec![0.0f64; n_variants * n_eval_days];
        
        for y_idx in 0..n_variants {
            for t_eval in 0..n_eval_days {
                let t_abs = eval_start_offset + t_eval;
                if t_abs >= max_history_days { continue; }
                
                let mut immunity_sum = 0.0f64;
                
                // Sum over all past variants
                for x_idx in 0..n_variants {
                    // Sum over history up to current time
                    for s in 0..=t_abs {
                        let freq = frequencies[x_idx * max_history_days + s];
                        if freq < 0.001 { continue; }
                        
                        let inc = landscape.daily_incidence[s];
                        if inc < 1.0 { continue; }
                        
                        // Load P_neut(x → y) from matrix
                        let p_neut = p_neut_matrix[x_idx * n_variants + y_idx] as f64;
                        
                        immunity_sum += freq as f64 * inc * p_neut;
                    }
                }
                
                immunity_matrix[y_idx * n_eval_days + t_eval] = immunity_sum;
            }
        }
        
        eprintln!("[PATH A] ✓ Immunity computed (single profile per variant-day)");
        
        // ═══════════════════════════════════════════════════════════════════
        // STEP 4: Compute gamma envelopes (reuse PATH B's approach)
        // ═══════════════════════════════════════════════════════════════════
        // For PATH A, we have a single immunity value (not 75 PK combos)
        // So gamma_min = gamma_max = gamma_mean = immunity_value
        
        let mut gamma_envelopes: Vec<Vec<(f64, f64, f64)>> = vec![vec![(0.0, 0.0, 0.0); n_eval_days]; n_variants];
        let mut weighted_avg_vec = vec![0.0f64; n_variants * n_eval_days];
        
        for y_idx in 0..n_variants {
            for t_idx in 0..n_eval_days {
                let immunity = immunity_matrix[y_idx * n_eval_days + t_idx];
                
                // For PATH A: min = max = mean = immunity (no PK variation)
                gamma_envelopes[y_idx][t_idx] = (immunity, immunity, immunity);
                
                // Weighted avg susceptibility (reuse PATH B formula)
                let mut weighted_sum = 0.0f64;
                let mut freq_sum = 0.0f64;
                
                let t_abs = eval_start_offset + t_idx;
                if t_abs < max_history_days {
                    for x_idx in 0..n_variants {
                        let freq = frequencies[x_idx * max_history_days + t_abs] as f64;
                        if freq >= 0.001 {
                            // Susceptibility = 1 - immunity (approximation)
                            let susceptibility = 1.0 - immunity.min(1.0);
                            weighted_sum += freq * susceptibility;
                            freq_sum += freq;
                        }
                    }
                }
                
                weighted_avg_vec[y_idx * n_eval_days + t_idx] = if freq_sum > 0.0 {
                    weighted_sum / freq_sum
                } else {
                    0.5  // Fallback
                };
            }
        }
        
        eprintln!("[PATH A] ✓ Gamma envelopes computed (deterministic - no PK variation)");
        
        // ═══════════════════════════════════════════════════════════════════
        // STEP 5: Build immunity_matrix_75pk for compatibility with existing code
        // ═══════════════════════════════════════════════════════════════════
        // PATH A doesn't use 75 PK combos, but we need this for the interface
        // Populate all 75 PK slots with the same immunity value
        
        let mut immunity_matrix_75pk: Vec<Vec<[f64; 75]>> = vec![vec![[0.0; 75]; n_eval_days]; n_variants];
        
        for y_idx in 0..n_variants {
            for t_idx in 0..n_eval_days {
                let immunity = immunity_matrix[y_idx * n_eval_days + t_idx];
                // All 75 PK combinations get the same value (no PK in PATH A)
                immunity_matrix_75pk[y_idx][t_idx] = [immunity; 75];
            }
        }
        
        // Build orig->sig mapping
        let mut orig_to_sig = vec![None; landscape.lineages.len()];
        for (sig_idx, &orig_idx) in significant_indices.iter().enumerate() {
            orig_to_sig[orig_idx] = Some(sig_idx);
        }
        
        let elapsed = start_time.elapsed();
        eprintln!("[ImmunityCache GPU PATH A] ✓ Built in {:.2}s", elapsed.as_secs_f64());
        
        let min_date = eval_start;
        let max_date = eval_end;
        
        eprintln!("[PATH A Complete] Immunity cache ready");
        eprintln!("  - immunity_matrix: {} variants × {} days (single profile)", n_variants, n_eval_days);
        eprintln!("  - gamma_envelopes: {} samples (deterministic)", n_variants * n_eval_days);
        eprintln!("  - Date range: {:?} to {:?}", min_date, max_date);
        
        Ok(Self {
            immunity_matrix_75pk,
            gamma_envelopes,
            population: landscape.population,
            start_date: eval_start,
            orig_to_sig,
            lineage_mutations: lineage_mutations.clone(),
            min_date,
            max_date,
            cutoff_used: eval_start,
        })
    }
