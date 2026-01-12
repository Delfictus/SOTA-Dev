# Current Session

> This file is the **primary handoff document** between Claude sessions.
> It preserves context and ensures perfect alignment with the vault.

---

## Session Status

| Field | Value |
|-------|-------|
| **Status** | GPU Wiring Complete - Ready for Benchmarks |
| **Target** | Download CryptoBench and apo-holo datasets |
| **Plan Reference** | GPU Wiring Plan Report |
| **Last Sync** | 2026-01-11T23:00:00Z |

---

## CRITICAL: GPU Wiring Completed

**Both sampling paths are NOW wired to real GPU kernels:**

| File | GPU Kernel | Status |
|------|------------|--------|
| `nova_path.rs` | PrismNova | ✅ WIRED |
| `amber_path.rs` | AmberMegaFusedHmc | ✅ WIRED |

### Verification Commands
```bash
# Verify no bail! statements (should return empty)
grep "bail!" crates/prism-validation/src/sampling/paths/nova_path.rs
grep "bail!" crates/prism-validation/src/sampling/paths/amber_path.rs

# Verify GPU kernel imports exist
grep "PrismNova" crates/prism-validation/src/sampling/paths/nova_path.rs
grep "AmberMegaFusedHmc" crates/prism-validation/src/sampling/paths/amber_path.rs
```

---

## Completed This Session (GPU Wiring Session)

### Phase A: GPU Wiring
- [x] **Phase A1**: Wired `nova_path.rs` to PrismNova GPU kernel
  - Added imports: `prism_gpu::prism_nova`, `prism_physics::amber_topology`, `cudarc`
  - Updated struct to hold `Arc<CudaContext>` and `Option<PrismNova>`
  - `load_structure()` now parses topology, creates PrismNova, uploads system data
  - `sample()` calls `nova.step()` in loop, collects real Betti numbers
  - Added topology conversion helper functions
  - Enforced Zero Fallback Policy (no CPU fallback)

- [x] **Phase A2**: Wired `amber_path.rs` to AmberMegaFusedHmc GPU kernel
  - Added imports: `prism_gpu::amber_mega_fused`, `build_exclusion_lists`
  - Updated struct to hold `Arc<CudaContext>` and `Option<AmberMegaFusedHmc>`
  - `load_structure()` builds topology tuples, uploads, runs minimization
  - `sample()` calls `hmc.run()` in loop, collects real conformations
  - Added topology conversion helper functions
  - Enforced Zero Fallback Policy (no CPU fallback)

### Documentation Updates
- [x] Updated `CLAUDE.md` with GPU Wiring Status section
- [x] Added verification commands to `CLAUDE.md`
- [x] Updated key files section to reflect WIRED status

---

## Next Subtasks (Phase B+C)

### Phase B: Data Download
- [ ] Clone CryptoBench repository (1107 structures)
  ```bash
  mkdir -p data/benchmarks/cryptobench
  git clone https://github.com/skrhakv/CryptoBench.git data/benchmarks/cryptobench_repo
  ```

- [ ] Download apo-holo pairs (15 pairs, 30 PDBs)
  ```bash
  mkdir -p data/benchmarks/apo_holo
  # Download from RCSB
  ```

### Phase C: Run Real Benchmarks
- [ ] Single structure test with real GPU sampling
- [ ] Apo-holo benchmark with real RMSD values
- [ ] CryptoBench ROC AUC (target: >0.70)

---

## Context Buffer

### Key Changes Made
```
- nova_path.rs: Was facade with bail!(), NOW wired to PrismNova
- amber_path.rs: Was facade with bail!(), NOW wired to AmberMegaFusedHmc
- Both use #[cfg(feature = "cryptic-gpu")] for conditional compilation
- Zero Fallback Policy enforced: no GPU = explicit error
```

### Important APIs Used
```
PrismNova:
- new(context, config) -> Result<Self>
- upload_system(&positions, &masses, &charges, &lj_params, &atom_types, &residue_atoms)
- step() -> Result<NovaStepResult>  (returns Betti numbers!)
- download_positions() -> Result<Vec<f32>>

AmberMegaFusedHmc:
- new(context, n_atoms) -> Result<Self>
- upload_topology(&positions, &bonds, &angles, &dihedrals, &nb_params, &exclusions)
- minimize(n_steps, step_size) -> Result<f32>
- initialize_velocities(temperature)
- run(n_steps, dt, temperature) -> Result<HmcRunResult>
- get_positions() -> Result<Vec<f32>>
```

### Blocking Issues
```
None currently. GPU wiring complete. Ready for benchmarks.
```

---

## Continuation Prompt

When starting next session:

```
Continue Phase 6 implementation.

GPU WIRING STATUS: COMPLETE
- nova_path.rs wired to PrismNova
- amber_path.rs wired to AmberMegaFusedHmc

NEXT: Download datasets and run real benchmarks
1. Clone CryptoBench (1107 structures)
2. Download apo-holo pairs (30 PDBs)
3. Run single structure test
4. Run apo-holo benchmark
5. Run CryptoBench benchmark (target: ROC AUC >0.70)
```

---

## Links

- [[Progress/implementation_status.json|Machine State]]
- [[Plans/Phase 6 Plan|Phase 6 Plan]]
- [[PRISM Dashboard|Dashboard]]

---

*End of Current Session. GPU Wiring Complete. Ready for Benchmarks.*
