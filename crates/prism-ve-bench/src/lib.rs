//! PRISM-VE VASIL Benchmark Library
//!
//! Exposes modules for testing

pub mod data_loader;
pub mod vasil_exact_metric;
pub mod fluxnet_vasil_adapter;
pub mod gpu_fluxnet_ve;

pub use vasil_exact_metric::{
    EnvelopeDecision, 
    DayDirection,
    VasilMetricComputer,
    CALIBRATED_IC50,
};
pub use fluxnet_vasil_adapter::{
    VEFluxNetState,
    VEFluxNetAction,
    VasilParameters,
    VEFluxNetOptimizer,
    DEFAULT_IC50,
};
