//! PRISM Runtime - World-Class Reactive Architecture
//!
//! A production-grade runtime system using actor-based concurrency,
//! reactive streams, and lock-free data structures.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                         EVENT BUS                                   │
//! │  (tokio::sync::broadcast - fan-out to all subscribers)             │
//! └─────────────────────────────────────────────────────────────────────┘
//!        │              │              │              │
//!        ▼              ▼              ▼              ▼
//!   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
//!   │Pipeline │   │  GPU    │   │Telemetry│   │   UI    │
//!   │ Actor   │   │ Actor   │   │ Actor   │   │ Actor   │
//!   └─────────┘   └─────────┘   └─────────┘   └─────────┘
//! ```
//!
//! # Key Features
//!
//! - **Actor Model**: Each subsystem runs independently with message passing
//! - **Event Sourcing**: All state changes are events (replayable, auditable)
//! - **Backpressure**: Bounded channels prevent memory exhaustion
//! - **Lock-Free**: Ring buffers for time-series data
//! - **Zero-Copy**: GPU memory mapping where possible

pub mod events;
pub mod state;
pub mod actors;
pub mod channels;

pub use events::{PrismEvent, EventBus};
pub use state::StateStore;
pub use actors::{ActorHandle, ActorSystem};
pub use channels::RingBuffer;

use anyhow::Result;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Runtime configuration
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// Event bus capacity (backpressure threshold)
    pub event_bus_capacity: usize,
    /// Ring buffer size for time-series data
    pub ring_buffer_size: usize,
    /// GPU telemetry polling interval (ms)
    pub gpu_poll_interval_ms: u64,
    /// Maximum concurrent actors
    pub max_actors: usize,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            event_bus_capacity: 1024,
            ring_buffer_size: 1000,
            gpu_poll_interval_ms: 100,
            max_actors: 16,
        }
    }
}

/// PRISM Runtime - orchestrates all actors and state
pub struct PrismRuntime {
    /// Shared application state
    pub state: Arc<StateStore>,
    /// Event bus for inter-actor communication
    pub event_bus: Arc<EventBus>,
    /// Actor system managing all actors
    actor_system: ActorSystem,
    /// Runtime configuration
    config: RuntimeConfig,
}

impl PrismRuntime {
    /// Create a new runtime instance
    pub fn new(config: RuntimeConfig) -> Result<Self> {
        let event_bus = Arc::new(EventBus::new(config.event_bus_capacity));
        let state = Arc::new(StateStore::new(config.ring_buffer_size));
        let actor_system = ActorSystem::new(config.max_actors);

        Ok(Self {
            state,
            event_bus,
            actor_system,
            config,
        })
    }

    /// Start the runtime with all actors
    pub async fn start(&mut self) -> Result<()> {
        log::info!("Starting PRISM Runtime with {} max actors", self.config.max_actors);

        // Spawn core actors
        self.actor_system.spawn_pipeline_actor(
            self.state.clone(),
            self.event_bus.clone(),
        ).await?;

        self.actor_system.spawn_gpu_actor(
            self.state.clone(),
            self.event_bus.clone(),
            self.config.gpu_poll_interval_ms,
        ).await?;

        self.actor_system.spawn_telemetry_actor(
            self.state.clone(),
            self.event_bus.clone(),
        ).await?;

        log::info!("PRISM Runtime started successfully");
        Ok(())
    }

    /// Shutdown the runtime gracefully
    pub async fn shutdown(mut self) -> Result<()> {
        log::info!("Shutting down PRISM Runtime...");
        self.actor_system.shutdown_all().await?;
        log::info!("PRISM Runtime shutdown complete");
        Ok(())
    }

    /// Get a receiver for UI updates
    pub fn subscribe(&self) -> tokio::sync::broadcast::Receiver<PrismEvent> {
        self.event_bus.subscribe()
    }

    /// Send a command to the runtime
    pub async fn send_command(&self, event: PrismEvent) -> Result<()> {
        self.event_bus.publish(event).await
    }

    /// Get current state snapshot (for UI rendering)
    pub async fn snapshot(&self) -> state::StateSnapshot {
        self.state.snapshot().await
    }
}
