// [STAGE-2-OPTIX] OptiX Dynamic Function Loader
//
// Loads OptiX functions dynamically from the NVIDIA driver using libloading.
// This implements the function table pattern used by OptiX.

use crate::error::{OptixError, Result};
use optix_sys::*;
use std::sync::OnceLock;

/// Global OptiX function table (initialized once)
static OPTIX_API: OnceLock<OptixApi> = OnceLock::new();

/// OptiX API function pointers
pub struct OptixApi {
    pub init: unsafe extern "C" fn() -> OptixResult,
    pub device_context_create: unsafe extern "C" fn(
        CUcontext,
        *const OptixDeviceContextOptions,
        *mut OptixDeviceContext,
    ) -> OptixResult,
    pub device_context_destroy: unsafe extern "C" fn(OptixDeviceContext) -> OptixResult,
    pub device_context_set_cache_enabled:
        unsafe extern "C" fn(OptixDeviceContext, i32) -> OptixResult,
    pub device_context_set_cache_location:
        unsafe extern "C" fn(OptixDeviceContext, *const i8) -> OptixResult,
}

impl OptixApi {
    /// Load OptiX functions from the driver
    unsafe fn load() -> Result<Self> {
        #[cfg(target_os = "linux")]
        let lib_name = "libnvoptix.so.1";
        #[cfg(target_os = "windows")]
        let lib_name = "nvoptix.dll";

        let lib = libloading::Library::new(lib_name).map_err(|e| {
            OptixError::InitializationFailed(format!("Failed to load OptiX library: {}", e))
        })?;

        // Load function pointers and copy them before leaking the library
        let init_fn: libloading::Symbol<unsafe extern "C" fn() -> OptixResult> = lib
            .get(b"optixInit\0")
            .map_err(|e| OptixError::InitializationFailed(format!("optixInit not found: {}", e)))?;
        let init_ptr = *init_fn;

        let create_fn: libloading::Symbol<
            unsafe extern "C" fn(
                CUcontext,
                *const OptixDeviceContextOptions,
                *mut OptixDeviceContext,
            ) -> OptixResult,
        > = lib.get(b"optixDeviceContextCreate\0").map_err(|e| {
            OptixError::InitializationFailed(format!("optixDeviceContextCreate not found: {}", e))
        })?;
        let create_ptr = *create_fn;

        let destroy_fn: libloading::Symbol<
            unsafe extern "C" fn(OptixDeviceContext) -> OptixResult,
        > = lib.get(b"optixDeviceContextDestroy\0").map_err(|e| {
            OptixError::InitializationFailed(format!("optixDeviceContextDestroy not found: {}", e))
        })?;
        let destroy_ptr = *destroy_fn;

        let cache_enabled_fn: libloading::Symbol<
            unsafe extern "C" fn(OptixDeviceContext, i32) -> OptixResult,
        > = lib.get(b"optixDeviceContextSetCacheEnabled\0").map_err(|e| {
            OptixError::InitializationFailed(format!(
                "optixDeviceContextSetCacheEnabled not found: {}",
                e
            ))
        })?;
        let cache_enabled_ptr = *cache_enabled_fn;

        let cache_location_fn: libloading::Symbol<
            unsafe extern "C" fn(OptixDeviceContext, *const i8) -> OptixResult,
        > = lib.get(b"optixDeviceContextSetCacheLocation\0").map_err(|e| {
            OptixError::InitializationFailed(format!(
                "optixDeviceContextSetCacheLocation not found: {}",
                e
            ))
        })?;
        let cache_location_ptr = *cache_location_fn;

        // Leak the library to keep it loaded for the program lifetime
        std::mem::forget(lib);

        Ok(Self {
            init: init_ptr,
            device_context_create: create_ptr,
            device_context_destroy: destroy_ptr,
            device_context_set_cache_enabled: cache_enabled_ptr,
            device_context_set_cache_location: cache_location_ptr,
        })
    }

    /// Get the global OptiX API (initializes on first call)
    pub fn get() -> Result<&'static OptixApi> {
        OPTIX_API
            .get_or_init(|| unsafe { Self::load().expect("Failed to load OptiX API") });
        OPTIX_API.get().ok_or_else(|| {
            OptixError::InitializationFailed("OptiX API not initialized".to_string())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires OptiX driver
    fn test_load_optix_api() {
        let api = OptixApi::get();
        assert!(api.is_ok(), "Failed to load OptiX API");
    }
}
