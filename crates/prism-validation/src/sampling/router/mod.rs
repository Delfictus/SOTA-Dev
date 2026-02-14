//! Routing Layer
//!
//! Provides the `HybridSampler` entry point that routes to appropriate backends.

mod hybrid_sampler;

pub use hybrid_sampler::{HybridSampler, RoutingStrategy};
