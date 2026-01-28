//! # VRAM Safety Guard - Sovereign GPU Memory Management
//!
//! Battle-tested memory safety for PRISM-Zero physics engine.
//! Prevents GPU driver crashes through proactive VRAM monitoring.
//!
//! ## Safety Protocol
//! - Query available GPU memory before allocation
//! - Enforce 90% threshold to prevent memory exhaustion
//! - Graceful panic with clear error messages
//! - Zero tolerance for driver crashes

use cudarc::driver::{CudaContext, DriverError};
use std::sync::Arc;
use thiserror::Error;

/// VRAM safety threshold - 90% of available memory
const VRAM_SAFETY_THRESHOLD: f64 = 0.9;

/// Memory allocation errors with sovereign context
#[derive(Error, Debug)]
pub enum VramGuardError {
    /// GPU memory query failed
    #[error("Failed to query GPU memory: {0}")]
    MemoryQueryFailed(#[from] DriverError),

    /// Insufficient VRAM available
    #[error("VRAM Exhaustion Risk: Required {required_mb}MB exceeds safe limit of {available_mb}MB (90% of {total_mb}MB)")]
    InsufficientVram {
        required_mb: u64,
        available_mb: u64,
        total_mb: u64,
    },

    /// GPU device not available
    #[error("GPU device unavailable for memory query")]
    DeviceUnavailable,
}

/// GPU memory information with sovereign metadata
#[derive(Debug, Clone)]
pub struct VramInfo {
    /// Total GPU memory in bytes
    pub total_bytes: usize,
    /// Available GPU memory in bytes
    pub free_bytes: usize,
    /// Used GPU memory in bytes
    pub used_bytes: usize,
    /// Safe allocation limit (90% of total)
    pub safe_limit_bytes: usize,
    /// GPU device ID
    pub device_id: i32,
}

impl VramInfo {
    /// Convert bytes to megabytes for human readability
    pub fn total_mb(&self) -> u64 {
        (self.total_bytes / (1024 * 1024)) as u64
    }

    /// Convert bytes to megabytes for human readability
    pub fn free_mb(&self) -> u64 {
        (self.free_bytes / (1024 * 1024)) as u64
    }

    /// Convert bytes to megabytes for human readability
    pub fn used_mb(&self) -> u64 {
        (self.used_bytes / (1024 * 1024)) as u64
    }

    /// Convert bytes to megabytes for human readability
    pub fn safe_limit_mb(&self) -> u64 {
        (self.safe_limit_bytes / (1024 * 1024)) as u64
    }

    /// Calculate memory utilization as percentage
    pub fn utilization_percent(&self) -> f64 {
        (self.used_bytes as f64 / self.total_bytes as f64) * 100.0
    }
}

/// VRAM Safety Guard - Sovereign GPU memory protection
#[derive(Debug)]
pub struct VramGuard {
    context: Arc<CudaContext>,
}

impl VramGuard {
    /// Create new VRAM guard for specified GPU context
    pub fn new(context: Arc<CudaContext>) -> Self {
        Self { context }
    }

    /// Query current GPU memory state with sovereign precision
    ///
    /// # Returns
    /// * `Result<VramInfo>` - Complete memory state or query error
    ///
    /// # Implementation
    /// Uses `cuMemGetInfo` from CUDA Driver API for authentic memory reporting
    pub fn query_vram(&self) -> Result<VramInfo, VramGuardError> {
        let mut free_bytes: usize = 0;
        let mut total_bytes: usize = 0;

        // SOVEREIGN CODE: Direct Hardware Query
        // We use unsafe here because we are in the 'prism-gpu' containment zone.
        // We MUST know the real memory state.
        unsafe {
            let result = cudarc::driver::sys::cuMemGetInfo_v2(
                &mut free_bytes as *mut usize,
                &mut total_bytes as *mut usize,
            );

            if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                // CUDA memory query failed - device unavailable or permission issue
                log::error!("CUDA cuMemGetInfo failed with code: {:?}", result);
                return Err(VramGuardError::DeviceUnavailable);
            }
        }

        // Calculate usage
        let used_bytes = total_bytes - free_bytes;
        let safe_limit_bytes = (total_bytes as f64 * VRAM_SAFETY_THRESHOLD) as usize;
        let device_id = 0; // Will be determined from context in future iteration

        // Log the reality (Telemetry)
        #[cfg(feature = "telemetry")]
        log::debug!(
            "🛡️ VRAM Status: {}/{} MB Free",
            free_bytes / (1024 * 1024),
            total_bytes / (1024 * 1024)
        );

        Ok(VramInfo {
            total_bytes,
            free_bytes,
            used_bytes,
            safe_limit_bytes,
            device_id,
        })
    }

    /// Verify sufficient VRAM for allocation with sovereign safety
    ///
    /// # Arguments
    /// * `required_bytes` - Memory allocation request in bytes
    ///
    /// # Returns
    /// * `Result<VramInfo>` - Memory state if safe, error if insufficient
    ///
    /// # Safety Protocol
    /// Enforces 90% threshold to prevent GPU driver crashes
    pub fn verify_allocation(&self, required_bytes: usize) -> Result<VramInfo, VramGuardError> {
        let vram_info = self.query_vram()?;

        // Check if allocation exceeds safe limit
        if required_bytes > vram_info.safe_limit_bytes {
            return Err(VramGuardError::InsufficientVram {
                required_mb: (required_bytes / (1024 * 1024)) as u64,
                available_mb: vram_info.safe_limit_mb(),
                total_mb: vram_info.total_mb(),
            });
        }

        // Check if allocation exceeds available memory
        if required_bytes > vram_info.free_bytes {
            return Err(VramGuardError::InsufficientVram {
                required_mb: (required_bytes / (1024 * 1024)) as u64,
                available_mb: vram_info.free_mb(),
                total_mb: vram_info.total_mb(),
            });
        }

        log::info!(
            "🛡️  VRAM Guard: Allocation approved - {}MB requested, {}MB available, {}MB total",
            (required_bytes / (1024 * 1024)),
            vram_info.free_mb(),
            vram_info.total_mb()
        );

        Ok(vram_info)
    }

    /// Perform comprehensive VRAM safety check before physics engine startup
    ///
    /// # Arguments
    /// * `trajectory_size_bytes` - Expected trajectory memory requirement
    /// * `workspace_size_bytes` - Physics engine workspace requirement
    ///
    /// # Returns
    /// * `Result<VramInfo>` - Memory state if safe for engine startup
    ///
    /// # Physics Engine Safety
    /// Validates total memory requirement for PIMC/NLNM solvers
    pub fn verify_physics_engine_startup(
        &self,
        trajectory_size_bytes: usize,
        workspace_size_bytes: usize,
    ) -> Result<VramInfo, VramGuardError> {
        let total_required = trajectory_size_bytes + workspace_size_bytes;

        log::info!(
            "🧬 Physics Engine Memory Requirements:\n\
             📊 Trajectory: {}MB\n\
             ⚙️  Workspace: {}MB\n\
             📈 Total: {}MB",
            trajectory_size_bytes / (1024 * 1024),
            workspace_size_bytes / (1024 * 1024),
            total_required / (1024 * 1024)
        );

        self.verify_allocation(total_required)
    }

    /// Emergency VRAM status for diagnostic reporting
    pub fn emergency_status(&self) -> String {
        match self.query_vram() {
            Ok(info) => {
                format!(
                    "🚨 VRAM Emergency Status:\n\
                     GPU: Device {}\n\
                     Total: {}MB\n\
                     Free: {}MB\n\
                     Used: {}MB ({}%)\n\
                     Safe Limit: {}MB",
                    info.device_id,
                    info.total_mb(),
                    info.free_mb(),
                    info.used_mb(),
                    info.utilization_percent(),
                    info.safe_limit_mb()
                )
            }
            Err(e) => {
                format!("🚨 VRAM Query Failed: {}", e)
            }
        }
    }
}

/// Global VRAM guard for convenient access across physics engine
static mut GLOBAL_VRAM_GUARD: Option<VramGuard> = None;
static INIT_GUARD: std::sync::Once = std::sync::Once::new();

/// Initialize global VRAM guard with specified GPU context
///
/// # Safety
/// Must be called once during application startup before any GPU operations
pub fn init_global_vram_guard(context: Arc<CudaContext>) {
    unsafe {
        INIT_GUARD.call_once(|| {
            GLOBAL_VRAM_GUARD = Some(VramGuard::new(context));
            log::info!("🛡️  Global VRAM Guard initialized");
        });
    }
}

/// Access global VRAM guard for memory verification
///
/// # Panics
/// If VRAM guard not initialized via `init_global_vram_guard`
pub fn global_vram_guard() -> &'static VramGuard {
    unsafe {
        GLOBAL_VRAM_GUARD.as_ref()
            .expect("VRAM Guard not initialized - call init_global_vram_guard() first")
    }
}

/// Convenient macro for VRAM verification in hot paths
///
/// # Usage
/// ```rust
/// ensure_vram!(required_bytes)?;
/// ```
#[macro_export]
macro_rules! ensure_vram {
    ($required_bytes:expr) => {
        $crate::memory::global_vram_guard().verify_allocation($required_bytes)
    };
}

/// Physics engine specific VRAM verification macro
///
/// # Usage
/// ```rust
/// ensure_physics_vram!(trajectory_size, workspace_size)?;
/// ```
#[macro_export]
macro_rules! ensure_physics_vram {
    ($trajectory_bytes:expr, $workspace_bytes:expr) => {
        $crate::memory::global_vram_guard()
            .verify_physics_engine_startup($trajectory_bytes, $workspace_bytes)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vram_info_calculations() {
        let info = VramInfo {
            total_bytes: 8 * 1024 * 1024 * 1024, // 8GB
            free_bytes: 6 * 1024 * 1024 * 1024,  // 6GB
            used_bytes: 2 * 1024 * 1024 * 1024,  // 2GB
            safe_limit_bytes: (8.0 * 1024.0 * 1024.0 * 1024.0 * 0.9) as usize, // 7.2GB
            device_id: 0,
        };

        assert_eq!(info.total_mb(), 8192);
        assert_eq!(info.free_mb(), 6144);
        assert_eq!(info.used_mb(), 2048);
        assert_eq!(info.safe_limit_mb(), 7372); // ~7.2GB in MB
        assert_eq!(info.utilization_percent(), 25.0);
    }

    #[test]
    fn test_vram_guard_error_display() {
        let error = VramGuardError::InsufficientVram {
            required_mb: 5000,
            available_mb: 4000,
            total_mb: 8000,
        };

        let error_msg = format!("{}", error);
        assert!(error_msg.contains("5000MB"));
        assert!(error_msg.contains("4000MB"));
        assert!(error_msg.contains("8000MB"));
    }
}