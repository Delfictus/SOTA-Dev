//! Migration Feature Flags (Strangler Pattern)
//!
//! Controls gradual rollout from AMBER (stable) to NOVA (greenfield) path.
//!
//! Migration stages:
//! 1. StableOnly - AMBER only, no NOVA
//! 2. Shadow - Both run, AMBER returns, divergence logged
//! 3. Canary(N%) - N% traffic to NOVA, rest to AMBER
//! 4. GreenfieldPrimary - NOVA primary, AMBER shadow
//! 5. GreenfieldOnly - NOVA only, AMBER removed

use serde::{Deserialize, Serialize};

/// Migration stage for gradual rollout
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MigrationStage {
    /// AMBER only - NOVA disabled entirely
    StableOnly,
    /// Both paths run, AMBER result returned, divergence logged
    Shadow,
    /// N% of traffic to NOVA (rest to AMBER)
    Canary(u8),
    /// NOVA primary, AMBER as shadow for validation
    GreenfieldPrimary,
    /// NOVA only - AMBER code path removed
    GreenfieldOnly,
}

impl Default for MigrationStage {
    fn default() -> Self {
        // Start with stable-only for safety
        MigrationStage::StableOnly
    }
}

impl std::fmt::Display for MigrationStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MigrationStage::StableOnly => write!(f, "StableOnly"),
            MigrationStage::Shadow => write!(f, "Shadow"),
            MigrationStage::Canary(pct) => write!(f, "Canary({}%)", pct),
            MigrationStage::GreenfieldPrimary => write!(f, "GreenfieldPrimary"),
            MigrationStage::GreenfieldOnly => write!(f, "GreenfieldOnly"),
        }
    }
}

/// Migration feature flags
#[derive(Debug, Clone)]
pub struct MigrationFlags {
    /// Current migration stage
    stage: MigrationStage,
    /// Counter for canary routing (used for deterministic % routing)
    request_counter: u64,
    /// Whether to log divergence metrics
    log_divergence: bool,
    /// Auto-rollback threshold (major divergence count)
    rollback_threshold: u32,
    /// Current major divergence count
    divergence_count: u32,
}

impl MigrationFlags {
    /// Create new migration flags at specified stage
    pub fn new(stage: MigrationStage) -> Self {
        Self {
            stage,
            request_counter: 0,
            log_divergence: true,
            rollback_threshold: 3,
            divergence_count: 0,
        }
    }

    /// Get current stage
    pub fn stage(&self) -> MigrationStage {
        self.stage
    }

    /// Check if NOVA should be used for this request
    pub fn should_use_nova(&mut self) -> bool {
        self.request_counter = self.request_counter.wrapping_add(1);

        match self.stage {
            MigrationStage::StableOnly => false,
            MigrationStage::Shadow => true, // Run both, but AMBER returns
            MigrationStage::Canary(pct) => {
                // Deterministic routing based on counter
                (self.request_counter % 100) < pct as u64
            }
            MigrationStage::GreenfieldPrimary => true,
            MigrationStage::GreenfieldOnly => true,
        }
    }

    /// Check if shadow comparison should run
    pub fn should_run_shadow(&self) -> bool {
        matches!(
            self.stage,
            MigrationStage::Shadow | MigrationStage::GreenfieldPrimary
        )
    }

    /// Alias for should_run_shadow (for API consistency)
    pub fn run_shadow(&self) -> bool {
        self.should_run_shadow()
    }

    /// Check if greenfield (NOVA) should be primary
    pub fn use_greenfield(&self) -> bool {
        matches!(
            self.stage,
            MigrationStage::GreenfieldPrimary | MigrationStage::GreenfieldOnly
        )
    }

    /// Check if AMBER result should be returned (vs NOVA)
    pub fn return_amber_result(&self) -> bool {
        matches!(
            self.stage,
            MigrationStage::StableOnly | MigrationStage::Shadow
        )
    }

    /// Record a divergence event
    pub fn record_divergence(&mut self, is_major: bool) {
        if is_major {
            self.divergence_count += 1;
            log::warn!(
                "Major divergence #{} (threshold: {})",
                self.divergence_count,
                self.rollback_threshold
            );
        }
    }

    /// Check if auto-rollback should trigger
    pub fn should_rollback(&self) -> bool {
        self.divergence_count >= self.rollback_threshold
    }

    /// Rollback to stable-only
    pub fn rollback(&mut self) {
        log::error!(
            "Auto-rollback triggered after {} major divergences",
            self.divergence_count
        );
        self.stage = MigrationStage::StableOnly;
        self.divergence_count = 0;
    }

    /// Advance to next migration stage
    pub fn advance(&mut self) -> bool {
        let next = match self.stage {
            MigrationStage::StableOnly => Some(MigrationStage::Shadow),
            MigrationStage::Shadow => Some(MigrationStage::Canary(10)),
            MigrationStage::Canary(pct) if pct < 50 => Some(MigrationStage::Canary(50)),
            MigrationStage::Canary(_) => Some(MigrationStage::GreenfieldPrimary),
            MigrationStage::GreenfieldPrimary => Some(MigrationStage::GreenfieldOnly),
            MigrationStage::GreenfieldOnly => None,
        };

        if let Some(stage) = next {
            log::info!("Migration: {} -> {}", self.stage, stage);
            self.stage = stage;
            self.divergence_count = 0;
            true
        } else {
            false
        }
    }

    /// Set divergence logging
    pub fn with_logging(mut self, enabled: bool) -> Self {
        self.log_divergence = enabled;
        self
    }

    /// Set rollback threshold
    pub fn with_rollback_threshold(mut self, threshold: u32) -> Self {
        self.rollback_threshold = threshold;
        self
    }

    /// Check if divergence logging is enabled
    pub fn logging_enabled(&self) -> bool {
        self.log_divergence
    }

    /// Get current divergence count
    pub fn divergence_count(&self) -> u32 {
        self.divergence_count
    }

    /// Get rollback threshold
    pub fn rollback_threshold(&self) -> u32 {
        self.rollback_threshold
    }
}

impl Default for MigrationFlags {
    fn default() -> Self {
        Self::new(MigrationStage::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage_display() {
        assert_eq!(format!("{}", MigrationStage::StableOnly), "StableOnly");
        assert_eq!(format!("{}", MigrationStage::Shadow), "Shadow");
        assert_eq!(format!("{}", MigrationStage::Canary(10)), "Canary(10%)");
        assert_eq!(
            format!("{}", MigrationStage::GreenfieldPrimary),
            "GreenfieldPrimary"
        );
        assert_eq!(
            format!("{}", MigrationStage::GreenfieldOnly),
            "GreenfieldOnly"
        );
    }

    #[test]
    fn test_stable_only_never_uses_nova() {
        let mut flags = MigrationFlags::new(MigrationStage::StableOnly);
        for _ in 0..100 {
            assert!(!flags.should_use_nova());
        }
        assert!(flags.return_amber_result());
        assert!(!flags.should_run_shadow());
    }

    #[test]
    fn test_shadow_runs_both() {
        let mut flags = MigrationFlags::new(MigrationStage::Shadow);
        assert!(flags.should_use_nova());
        assert!(flags.return_amber_result()); // But returns AMBER
        assert!(flags.should_run_shadow());
    }

    #[test]
    fn test_canary_routing() {
        let mut flags = MigrationFlags::new(MigrationStage::Canary(50));

        // Count NOVA selections over 100 requests
        let nova_count: u64 = (0..100).filter(|_| flags.should_use_nova()).count() as u64;

        // Should be roughly 50%
        assert!(
            nova_count >= 40 && nova_count <= 60,
            "Expected ~50 NOVA, got {}",
            nova_count
        );
    }

    #[test]
    fn test_greenfield_primary() {
        let mut flags = MigrationFlags::new(MigrationStage::GreenfieldPrimary);
        assert!(flags.should_use_nova());
        assert!(!flags.return_amber_result()); // Returns NOVA
        assert!(flags.should_run_shadow()); // Still compares
    }

    #[test]
    fn test_greenfield_only() {
        let mut flags = MigrationFlags::new(MigrationStage::GreenfieldOnly);
        assert!(flags.should_use_nova());
        assert!(!flags.return_amber_result());
        assert!(!flags.should_run_shadow()); // No shadow
    }

    #[test]
    fn test_auto_rollback() {
        let mut flags = MigrationFlags::new(MigrationStage::Shadow).with_rollback_threshold(3);

        assert!(!flags.should_rollback());
        flags.record_divergence(true);
        flags.record_divergence(true);
        assert!(!flags.should_rollback());
        flags.record_divergence(true);
        assert!(flags.should_rollback());

        flags.rollback();
        assert_eq!(flags.stage(), MigrationStage::StableOnly);
        assert_eq!(flags.divergence_count(), 0);
    }

    #[test]
    fn test_migration_advancement() {
        let mut flags = MigrationFlags::new(MigrationStage::StableOnly);

        assert!(flags.advance());
        assert_eq!(flags.stage(), MigrationStage::Shadow);

        assert!(flags.advance());
        assert_eq!(flags.stage(), MigrationStage::Canary(10));

        assert!(flags.advance());
        assert_eq!(flags.stage(), MigrationStage::Canary(50));

        assert!(flags.advance());
        assert_eq!(flags.stage(), MigrationStage::GreenfieldPrimary);

        assert!(flags.advance());
        assert_eq!(flags.stage(), MigrationStage::GreenfieldOnly);

        // Can't advance beyond GreenfieldOnly
        assert!(!flags.advance());
        assert_eq!(flags.stage(), MigrationStage::GreenfieldOnly);
    }

    #[test]
    fn test_minor_divergence_no_rollback() {
        let mut flags = MigrationFlags::new(MigrationStage::Shadow).with_rollback_threshold(3);

        // Minor divergences don't count toward rollback
        for _ in 0..10 {
            flags.record_divergence(false);
        }
        assert!(!flags.should_rollback());
    }

    #[test]
    fn test_default() {
        let flags = MigrationFlags::default();
        assert_eq!(flags.stage(), MigrationStage::StableOnly);
    }
}
