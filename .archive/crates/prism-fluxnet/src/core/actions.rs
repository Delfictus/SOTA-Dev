//! Universal action space for FluxNet RL.
//!
//! Defines actions that can be taken in each phase to adjust parameters.
//!
//! Implements PRISM GPU Plan §3.2: UniversalAction.

use serde::{Deserialize, Serialize};

/// Universal action enum covering all 7 phases.
///
/// Each phase has its own action sub-enum with phase-specific adjustments.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum UniversalAction {
    /// Phase 0: Dendritic Reservoir actions
    Phase0(DendriticAction),
    /// Phase 1: Active Inference actions
    Phase1(ActiveInferenceAction),
    /// Phase 2: Thermodynamic actions
    Phase2(ThermodynamicAction),
    /// Phase 3: Quantum-Classical actions
    Phase3(QuantumAction),
    /// Phases 4/5: Geodesic actions
    Phase4(GeodesicAction),
    /// Phase 6: TDA actions
    Phase6(TDAAction),
    /// Phase 7: Ensemble actions
    Phase7(EnsembleAction),
    /// Warmstart configuration actions
    Warmstart(WarmstartAction),
    /// Memetic algorithm actions
    Memetic(MemeticAction),
    /// Geometry coupling actions (metaphysical telemetry)
    Geometry(GeometryAction),
    /// MEC (Molecular Emergent Computing) actions
    MEC(MecAction),
    /// CMA-ES optimization actions
    CMA(CmaAction),
    /// No-op (do nothing)
    NoOp,
}

impl UniversalAction {
    /// Converts the action to a unique integer index for Q-table lookup.
    ///
    /// ## Index Allocation (104 total actions)
    /// - Phase0: 0-7 (8 actions)
    /// - Phase1: 8-15 (8 actions)
    /// - Phase2: 16-23 (8 actions)
    /// - Phase3: 24-31 (8 actions)
    /// - Phase4: 32-39 (8 actions)
    /// - Phase6: 40-47 (8 actions)
    /// - Phase7: 48-55 (8 actions)
    /// - Warmstart: 56-63 (8 actions)
    /// - Memetic: 64-71 (8 actions)
    /// - Geometry: 72-79 (8 actions)
    /// - MEC: 80-87 (8 actions)
    /// - CMA: 88-95 (8 actions)
    /// - NoOp: 103
    pub fn to_index(&self) -> usize {
        match self {
            UniversalAction::Phase0(action) => action.to_index(),
            UniversalAction::Phase1(action) => 8 + action.to_index(),
            UniversalAction::Phase2(action) => 16 + action.to_index(),
            UniversalAction::Phase3(action) => 24 + action.to_index(),
            UniversalAction::Phase4(action) => 32 + action.to_index(),
            UniversalAction::Phase6(action) => 40 + action.to_index(),
            UniversalAction::Phase7(action) => 48 + action.to_index(),
            UniversalAction::Warmstart(action) => 56 + action.to_index(),
            UniversalAction::Memetic(action) => 64 + action.to_index(),
            UniversalAction::Geometry(action) => 72 + action.to_index(),
            UniversalAction::MEC(action) => 80 + action.to_index(),
            UniversalAction::CMA(action) => 88 + action.to_index(),
            UniversalAction::NoOp => 103,
        }
    }

    /// Creates an action from an index (inverse of `to_index`).
    pub fn from_index(index: usize, phase: &str) -> Option<Self> {
        match phase {
            "Phase0-DendriticReservoir" => {
                DendriticAction::from_index(index).map(UniversalAction::Phase0)
            }
            "Phase1-ActiveInference" => {
                ActiveInferenceAction::from_index(index - 8).map(UniversalAction::Phase1)
            }
            "Phase2-Thermodynamic" => {
                ThermodynamicAction::from_index(index - 16).map(UniversalAction::Phase2)
            }
            "Phase3-QuantumClassical" => {
                QuantumAction::from_index(index - 24).map(UniversalAction::Phase3)
            }
            "Phase4-Geodesic" | "Phase5-NetworkTopology" => {
                GeodesicAction::from_index(index - 32).map(UniversalAction::Phase4)
            }
            "Phase6-TDA" => TDAAction::from_index(index - 40).map(UniversalAction::Phase6),
            "Phase7-Ensemble" => {
                EnsembleAction::from_index(index - 48).map(UniversalAction::Phase7)
            }
            "Warmstart" => WarmstartAction::from_index(index - 56).map(UniversalAction::Warmstart),
            "Memetic" => MemeticAction::from_index(index - 64).map(UniversalAction::Memetic),
            "Geometry" => GeometryAction::from_index(index - 72).map(UniversalAction::Geometry),
            "MEC" | "PhaseM-MEC" => MecAction::from_index(index - 80).map(UniversalAction::MEC),
            "CMA" | "PhaseX-CMA" => CmaAction::from_index(index - 88).map(UniversalAction::CMA),
            _ => None,
        }
    }

    /// Returns all valid actions for a given phase.
    pub fn all_actions_for_phase(phase: &str) -> Vec<Self> {
        match phase {
            "Phase0-DendriticReservoir" => DendriticAction::all()
                .into_iter()
                .map(UniversalAction::Phase0)
                .collect(),
            "Phase1-ActiveInference" => ActiveInferenceAction::all()
                .into_iter()
                .map(UniversalAction::Phase1)
                .collect(),
            "Phase2-Thermodynamic" => ThermodynamicAction::all()
                .into_iter()
                .map(UniversalAction::Phase2)
                .collect(),
            "Phase3-QuantumClassical" => QuantumAction::all()
                .into_iter()
                .map(UniversalAction::Phase3)
                .collect(),
            "Phase4-Geodesic" | "Phase5-NetworkTopology" => GeodesicAction::all()
                .into_iter()
                .map(UniversalAction::Phase4)
                .collect(),
            "Phase6-TDA" => TDAAction::all()
                .into_iter()
                .map(UniversalAction::Phase6)
                .collect(),
            "Phase7-Ensemble" => EnsembleAction::all()
                .into_iter()
                .map(UniversalAction::Phase7)
                .collect(),
            "Warmstart" => WarmstartAction::all()
                .into_iter()
                .map(UniversalAction::Warmstart)
                .collect(),
            "Memetic" => MemeticAction::all()
                .into_iter()
                .map(UniversalAction::Memetic)
                .collect(),
            "Geometry" => GeometryAction::all()
                .into_iter()
                .map(UniversalAction::Geometry)
                .collect(),
            "MEC" | "PhaseM-MEC" => MecAction::all()
                .into_iter()
                .map(UniversalAction::MEC)
                .collect(),
            "CMA" | "PhaseX-CMA" => CmaAction::all()
                .into_iter()
                .map(UniversalAction::CMA)
                .collect(),
            _ => vec![UniversalAction::NoOp],
        }
    }
}

// Phase 0: Dendritic Reservoir Actions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DendriticAction {
    IncreaseLearnRate,
    DecreaseLearnRate,
    IncreaseSparsity,
    DecreaseSparsity,
    SwitchIntegrationSum,
    SwitchIntegrationMax,
    SwitchIntegrationGated,
    NoOp,
}

impl DendriticAction {
    fn to_index(&self) -> usize {
        match self {
            Self::IncreaseLearnRate => 0,
            Self::DecreaseLearnRate => 1,
            Self::IncreaseSparsity => 2,
            Self::DecreaseSparsity => 3,
            Self::SwitchIntegrationSum => 4,
            Self::SwitchIntegrationMax => 5,
            Self::SwitchIntegrationGated => 6,
            Self::NoOp => 7,
        }
    }

    fn from_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(Self::IncreaseLearnRate),
            1 => Some(Self::DecreaseLearnRate),
            2 => Some(Self::IncreaseSparsity),
            3 => Some(Self::DecreaseSparsity),
            4 => Some(Self::SwitchIntegrationSum),
            5 => Some(Self::SwitchIntegrationMax),
            6 => Some(Self::SwitchIntegrationGated),
            7 => Some(Self::NoOp),
            _ => None,
        }
    }

    fn all() -> Vec<Self> {
        vec![
            Self::IncreaseLearnRate,
            Self::DecreaseLearnRate,
            Self::IncreaseSparsity,
            Self::DecreaseSparsity,
            Self::SwitchIntegrationSum,
            Self::SwitchIntegrationMax,
            Self::SwitchIntegrationGated,
            Self::NoOp,
        ]
    }
}

// Phase 1: Active Inference Actions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ActiveInferenceAction {
    IncreasePrecision,
    DecreasePrecision,
    IncreaseExploration,
    DecreaseExploration,
    SwitchPolicyGreedy,
    SwitchPolicyThompson,
    UpdateBeliefs,
    NoOp,
}

impl ActiveInferenceAction {
    fn to_index(&self) -> usize {
        match self {
            Self::IncreasePrecision => 0,
            Self::DecreasePrecision => 1,
            Self::IncreaseExploration => 2,
            Self::DecreaseExploration => 3,
            Self::SwitchPolicyGreedy => 4,
            Self::SwitchPolicyThompson => 5,
            Self::UpdateBeliefs => 6,
            Self::NoOp => 7,
        }
    }

    fn from_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(Self::IncreasePrecision),
            1 => Some(Self::DecreasePrecision),
            2 => Some(Self::IncreaseExploration),
            3 => Some(Self::DecreaseExploration),
            4 => Some(Self::SwitchPolicyGreedy),
            5 => Some(Self::SwitchPolicyThompson),
            6 => Some(Self::UpdateBeliefs),
            7 => Some(Self::NoOp),
            _ => None,
        }
    }

    fn all() -> Vec<Self> {
        vec![
            Self::IncreasePrecision,
            Self::DecreasePrecision,
            Self::IncreaseExploration,
            Self::DecreaseExploration,
            Self::SwitchPolicyGreedy,
            Self::SwitchPolicyThompson,
            Self::UpdateBeliefs,
            Self::NoOp,
        ]
    }
}

// Phase 2: Thermodynamic Actions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ThermodynamicAction {
    IncreaseTemperature,
    DecreaseTemperature,
    IncreaseCoolingRate,
    DecreaseCoolingRate,
    AnnealFast,
    AnnealSlow,
    Reheat,
    NoOp,
}

impl ThermodynamicAction {
    fn to_index(&self) -> usize {
        match self {
            Self::IncreaseTemperature => 0,
            Self::DecreaseTemperature => 1,
            Self::IncreaseCoolingRate => 2,
            Self::DecreaseCoolingRate => 3,
            Self::AnnealFast => 4,
            Self::AnnealSlow => 5,
            Self::Reheat => 6,
            Self::NoOp => 7,
        }
    }

    fn from_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(Self::IncreaseTemperature),
            1 => Some(Self::DecreaseTemperature),
            2 => Some(Self::IncreaseCoolingRate),
            3 => Some(Self::DecreaseCoolingRate),
            4 => Some(Self::AnnealFast),
            5 => Some(Self::AnnealSlow),
            6 => Some(Self::Reheat),
            7 => Some(Self::NoOp),
            _ => None,
        }
    }

    fn all() -> Vec<Self> {
        vec![
            Self::IncreaseTemperature,
            Self::DecreaseTemperature,
            Self::IncreaseCoolingRate,
            Self::DecreaseCoolingRate,
            Self::AnnealFast,
            Self::AnnealSlow,
            Self::Reheat,
            Self::NoOp,
        ]
    }
}

// Phase 3: Quantum Actions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum QuantumAction {
    IncreaseEvolutionTime,
    DecreaseEvolutionTime,
    IncreaseCouplingStrength,
    DecreaseCouplingStrength,
    ApplyHadamard,
    ApplyPhaseGate,
    MeasureState,
    NoOp,
}

impl QuantumAction {
    fn to_index(&self) -> usize {
        match self {
            Self::IncreaseEvolutionTime => 0,
            Self::DecreaseEvolutionTime => 1,
            Self::IncreaseCouplingStrength => 2,
            Self::DecreaseCouplingStrength => 3,
            Self::ApplyHadamard => 4,
            Self::ApplyPhaseGate => 5,
            Self::MeasureState => 6,
            Self::NoOp => 7,
        }
    }

    fn from_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(Self::IncreaseEvolutionTime),
            1 => Some(Self::DecreaseEvolutionTime),
            2 => Some(Self::IncreaseCouplingStrength),
            3 => Some(Self::DecreaseCouplingStrength),
            4 => Some(Self::ApplyHadamard),
            5 => Some(Self::ApplyPhaseGate),
            6 => Some(Self::MeasureState),
            7 => Some(Self::NoOp),
            _ => None,
        }
    }

    fn all() -> Vec<Self> {
        vec![
            Self::IncreaseEvolutionTime,
            Self::DecreaseEvolutionTime,
            Self::IncreaseCouplingStrength,
            Self::DecreaseCouplingStrength,
            Self::ApplyHadamard,
            Self::ApplyPhaseGate,
            Self::MeasureState,
            Self::NoOp,
        ]
    }
}

// Phase 4/5: Geodesic Actions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GeodesicAction {
    PrioritizeHighCentrality,
    PrioritizeLowCentrality,
    PrioritizeHighEccentricity,
    PrioritizeLowEccentricity,
    UseFloydWarshall,
    UseDijkstra,
    CacheDistances,
    NoOp,
}

impl GeodesicAction {
    fn to_index(&self) -> usize {
        match self {
            Self::PrioritizeHighCentrality => 0,
            Self::PrioritizeLowCentrality => 1,
            Self::PrioritizeHighEccentricity => 2,
            Self::PrioritizeLowEccentricity => 3,
            Self::UseFloydWarshall => 4,
            Self::UseDijkstra => 5,
            Self::CacheDistances => 6,
            Self::NoOp => 7,
        }
    }

    fn from_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(Self::PrioritizeHighCentrality),
            1 => Some(Self::PrioritizeLowCentrality),
            2 => Some(Self::PrioritizeHighEccentricity),
            3 => Some(Self::PrioritizeLowEccentricity),
            4 => Some(Self::UseFloydWarshall),
            5 => Some(Self::UseDijkstra),
            6 => Some(Self::CacheDistances),
            7 => Some(Self::NoOp),
            _ => None,
        }
    }

    fn all() -> Vec<Self> {
        vec![
            Self::PrioritizeHighCentrality,
            Self::PrioritizeLowCentrality,
            Self::PrioritizeHighEccentricity,
            Self::PrioritizeLowEccentricity,
            Self::UseFloydWarshall,
            Self::UseDijkstra,
            Self::CacheDistances,
            Self::NoOp,
        ]
    }
}

// Phase 6: TDA Actions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TDAAction {
    IncreaseFilterationResolution,
    DecreaseFilterationResolution,
    ComputeBetti0,
    ComputeBetti1,
    PrioritizePersistentFeatures,
    PrioritizeTransientFeatures,
    RebuildComplex,
    NoOp,
}

impl TDAAction {
    fn to_index(&self) -> usize {
        match self {
            Self::IncreaseFilterationResolution => 0,
            Self::DecreaseFilterationResolution => 1,
            Self::ComputeBetti0 => 2,
            Self::ComputeBetti1 => 3,
            Self::PrioritizePersistentFeatures => 4,
            Self::PrioritizeTransientFeatures => 5,
            Self::RebuildComplex => 6,
            Self::NoOp => 7,
        }
    }

    fn from_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(Self::IncreaseFilterationResolution),
            1 => Some(Self::DecreaseFilterationResolution),
            2 => Some(Self::ComputeBetti0),
            3 => Some(Self::ComputeBetti1),
            4 => Some(Self::PrioritizePersistentFeatures),
            5 => Some(Self::PrioritizeTransientFeatures),
            6 => Some(Self::RebuildComplex),
            7 => Some(Self::NoOp),
            _ => None,
        }
    }

    fn all() -> Vec<Self> {
        vec![
            Self::IncreaseFilterationResolution,
            Self::DecreaseFilterationResolution,
            Self::ComputeBetti0,
            Self::ComputeBetti1,
            Self::PrioritizePersistentFeatures,
            Self::PrioritizeTransientFeatures,
            Self::RebuildComplex,
            Self::NoOp,
        ]
    }
}

// Phase 7: Ensemble Actions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EnsembleAction {
    IncreaseReplicas,
    DecreaseReplicas,
    IncreaseDiversityWeight,
    DecreaseDiversityWeight,
    VoteMajority,
    VoteWeighted,
    Rerank,
    NoOp,
}

impl EnsembleAction {
    fn to_index(&self) -> usize {
        match self {
            Self::IncreaseReplicas => 0,
            Self::DecreaseReplicas => 1,
            Self::IncreaseDiversityWeight => 2,
            Self::DecreaseDiversityWeight => 3,
            Self::VoteMajority => 4,
            Self::VoteWeighted => 5,
            Self::Rerank => 6,
            Self::NoOp => 7,
        }
    }

    fn from_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(Self::IncreaseReplicas),
            1 => Some(Self::DecreaseReplicas),
            2 => Some(Self::IncreaseDiversityWeight),
            3 => Some(Self::DecreaseDiversityWeight),
            4 => Some(Self::VoteMajority),
            5 => Some(Self::VoteWeighted),
            6 => Some(Self::Rerank),
            7 => Some(Self::NoOp),
            _ => None,
        }
    }

    fn all() -> Vec<Self> {
        vec![
            Self::IncreaseReplicas,
            Self::DecreaseReplicas,
            Self::IncreaseDiversityWeight,
            Self::DecreaseDiversityWeight,
            Self::VoteMajority,
            Self::VoteWeighted,
            Self::Rerank,
            Self::NoOp,
        ]
    }
}

// Warmstart Configuration Actions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum WarmstartAction {
    EnableStructuralAnchors,
    DisableStructuralAnchors,
    EnableTopologicalAnchors,
    DisableTopologicalAnchors,
    IncreaseAnchorFraction,
    DecreaseAnchorFraction,
    IncreaseMaxColors,
    NoOp,
}

impl WarmstartAction {
    fn to_index(&self) -> usize {
        match self {
            Self::EnableStructuralAnchors => 0,
            Self::DisableStructuralAnchors => 1,
            Self::EnableTopologicalAnchors => 2,
            Self::DisableTopologicalAnchors => 3,
            Self::IncreaseAnchorFraction => 4,
            Self::DecreaseAnchorFraction => 5,
            Self::IncreaseMaxColors => 6,
            Self::NoOp => 7,
        }
    }

    fn from_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(Self::EnableStructuralAnchors),
            1 => Some(Self::DisableStructuralAnchors),
            2 => Some(Self::EnableTopologicalAnchors),
            3 => Some(Self::DisableTopologicalAnchors),
            4 => Some(Self::IncreaseAnchorFraction),
            5 => Some(Self::DecreaseAnchorFraction),
            6 => Some(Self::IncreaseMaxColors),
            7 => Some(Self::NoOp),
            _ => None,
        }
    }

    fn all() -> Vec<Self> {
        vec![
            Self::EnableStructuralAnchors,
            Self::DisableStructuralAnchors,
            Self::EnableTopologicalAnchors,
            Self::DisableTopologicalAnchors,
            Self::IncreaseAnchorFraction,
            Self::DecreaseAnchorFraction,
            Self::IncreaseMaxColors,
            Self::NoOp,
        ]
    }
}

// Memetic Algorithm Actions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MemeticAction {
    IncreaseMutationRate,
    DecreaseMutationRate,
    IncreaseCrossoverRate,
    DecreaseCrossoverRate,
    IncreaseLocalSearchIterations,
    DecreaseLocalSearchIterations,
    EnableEarlyTermination,
    NoOp,
}

impl MemeticAction {
    fn to_index(&self) -> usize {
        match self {
            Self::IncreaseMutationRate => 0,
            Self::DecreaseMutationRate => 1,
            Self::IncreaseCrossoverRate => 2,
            Self::DecreaseCrossoverRate => 3,
            Self::IncreaseLocalSearchIterations => 4,
            Self::DecreaseLocalSearchIterations => 5,
            Self::EnableEarlyTermination => 6,
            Self::NoOp => 7,
        }
    }

    fn from_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(Self::IncreaseMutationRate),
            1 => Some(Self::DecreaseMutationRate),
            2 => Some(Self::IncreaseCrossoverRate),
            3 => Some(Self::DecreaseCrossoverRate),
            4 => Some(Self::IncreaseLocalSearchIterations),
            5 => Some(Self::DecreaseLocalSearchIterations),
            6 => Some(Self::EnableEarlyTermination),
            7 => Some(Self::NoOp),
            _ => None,
        }
    }

    fn all() -> Vec<Self> {
        vec![
            Self::IncreaseMutationRate,
            Self::DecreaseMutationRate,
            Self::IncreaseCrossoverRate,
            Self::DecreaseCrossoverRate,
            Self::IncreaseLocalSearchIterations,
            Self::DecreaseLocalSearchIterations,
            Self::EnableEarlyTermination,
            Self::NoOp,
        ]
    }
}

// MEC (Molecular Emergent Computing) Actions
/// Actions for adjusting molecular simulation parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MecAction {
    /// Increase molecular dynamics temperature
    IncreaseTemperature,
    /// Decrease molecular dynamics temperature
    DecreaseTemperature,
    /// Increase simulation timestep
    IncreaseTimestep,
    /// Decrease simulation timestep
    DecreaseTimestep,
    /// Increase reaction rate constants
    IncreaseReactionRates,
    /// Decrease reaction rate constants
    DecreaseReactionRates,
    /// Enable GPU acceleration
    EnableGPU,
    /// No operation
    NoOp,
}

impl MecAction {
    fn to_index(&self) -> usize {
        match self {
            Self::IncreaseTemperature => 0,
            Self::DecreaseTemperature => 1,
            Self::IncreaseTimestep => 2,
            Self::DecreaseTimestep => 3,
            Self::IncreaseReactionRates => 4,
            Self::DecreaseReactionRates => 5,
            Self::EnableGPU => 6,
            Self::NoOp => 7,
        }
    }

    fn from_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(Self::IncreaseTemperature),
            1 => Some(Self::DecreaseTemperature),
            2 => Some(Self::IncreaseTimestep),
            3 => Some(Self::DecreaseTimestep),
            4 => Some(Self::IncreaseReactionRates),
            5 => Some(Self::DecreaseReactionRates),
            6 => Some(Self::EnableGPU),
            7 => Some(Self::NoOp),
            _ => None,
        }
    }

    fn all() -> Vec<Self> {
        vec![
            Self::IncreaseTemperature,
            Self::DecreaseTemperature,
            Self::IncreaseTimestep,
            Self::DecreaseTimestep,
            Self::IncreaseReactionRates,
            Self::DecreaseReactionRates,
            Self::EnableGPU,
            Self::NoOp,
        ]
    }
}

// CMA-ES Optimization Actions
/// Actions for adjusting CMA-ES transfer entropy optimization.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CmaAction {
    /// Increase population size (λ)
    IncreasePopulationSize,
    /// Decrease population size
    DecreasePopulationSize,
    /// Increase step size (σ)
    IncreaseStepSize,
    /// Decrease step size
    DecreaseStepSize,
    /// Increase damping factor
    IncreaseDamping,
    /// Decrease damping factor
    DecreaseDamping,
    /// Reset covariance matrix
    ResetCovariance,
    /// No operation
    NoOp,
}

impl CmaAction {
    fn to_index(&self) -> usize {
        match self {
            Self::IncreasePopulationSize => 0,
            Self::DecreasePopulationSize => 1,
            Self::IncreaseStepSize => 2,
            Self::DecreaseStepSize => 3,
            Self::IncreaseDamping => 4,
            Self::DecreaseDamping => 5,
            Self::ResetCovariance => 6,
            Self::NoOp => 7,
        }
    }

    fn from_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(Self::IncreasePopulationSize),
            1 => Some(Self::DecreasePopulationSize),
            2 => Some(Self::IncreaseStepSize),
            3 => Some(Self::DecreaseStepSize),
            4 => Some(Self::IncreaseDamping),
            5 => Some(Self::DecreaseDamping),
            6 => Some(Self::ResetCovariance),
            7 => Some(Self::NoOp),
            _ => None,
        }
    }

    fn all() -> Vec<Self> {
        vec![
            Self::IncreasePopulationSize,
            Self::DecreasePopulationSize,
            Self::IncreaseStepSize,
            Self::DecreaseStepSize,
            Self::IncreaseDamping,
            Self::DecreaseDamping,
            Self::ResetCovariance,
            Self::NoOp,
        ]
    }
}

// Geometry Coupling Actions (Metaphysical Telemetry Feedback)
/// Actions for responding to geometric stress telemetry.
///
/// These actions adjust phase parameters based on geometric stress levels
/// from Phase 4 (Geodesic) and Phase 6 (TDA). Implements metaphysical coupling
/// where geometry telemetry influences all phases.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GeometryAction {
    /// Increase Phase 1 active inference exploration (higher prediction error)
    IncreaseActiveInferenceExploration,
    /// Decrease Phase 1 exploration
    DecreaseActiveInferenceExploration,
    /// Trigger Phase 2 thermodynamic reheat (raise temperature)
    ReheatThermodynamic,
    /// Cool Phase 2 faster (lower temperature)
    CoolThermodynamic,
    /// Increase Phase 3 quantum coupling strength
    IncreaseQuantumCoupling,
    /// Decrease Phase 3 quantum coupling strength
    DecreaseQuantumCoupling,
    /// Intensify Phase 7 memetic local search in hotspot regions
    IntensifyMemeticSearch,
    /// NoOp (no geometry-based adjustment)
    NoOp,
}

impl GeometryAction {
    fn to_index(&self) -> usize {
        match self {
            Self::IncreaseActiveInferenceExploration => 0,
            Self::DecreaseActiveInferenceExploration => 1,
            Self::ReheatThermodynamic => 2,
            Self::CoolThermodynamic => 3,
            Self::IncreaseQuantumCoupling => 4,
            Self::DecreaseQuantumCoupling => 5,
            Self::IntensifyMemeticSearch => 6,
            Self::NoOp => 7,
        }
    }

    fn from_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(Self::IncreaseActiveInferenceExploration),
            1 => Some(Self::DecreaseActiveInferenceExploration),
            2 => Some(Self::ReheatThermodynamic),
            3 => Some(Self::CoolThermodynamic),
            4 => Some(Self::IncreaseQuantumCoupling),
            5 => Some(Self::DecreaseQuantumCoupling),
            6 => Some(Self::IntensifyMemeticSearch),
            7 => Some(Self::NoOp),
            _ => None,
        }
    }

    fn all() -> Vec<Self> {
        vec![
            Self::IncreaseActiveInferenceExploration,
            Self::DecreaseActiveInferenceExploration,
            Self::ReheatThermodynamic,
            Self::CoolThermodynamic,
            Self::IncreaseQuantumCoupling,
            Self::DecreaseQuantumCoupling,
            Self::IntensifyMemeticSearch,
            Self::NoOp,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_indexing() {
        let action = UniversalAction::Phase0(DendriticAction::IncreaseLearnRate);
        assert_eq!(action.to_index(), 0);

        let action = UniversalAction::Phase2(ThermodynamicAction::IncreaseTemperature);
        assert_eq!(action.to_index(), 16);

        let action = UniversalAction::Geometry(GeometryAction::ReheatThermodynamic);
        assert_eq!(action.to_index(), 74); // 72 + 2

        let action = UniversalAction::NoOp;
        assert_eq!(action.to_index(), 103);
    }

    #[test]
    fn test_action_round_trip() {
        let original = UniversalAction::Phase0(DendriticAction::SwitchIntegrationGated);
        let index = original.to_index();
        let reconstructed =
            UniversalAction::from_index(index, "Phase0-DendriticReservoir").unwrap();
        assert_eq!(original, reconstructed);
    }

    #[test]
    fn test_all_actions() {
        let actions = UniversalAction::all_actions_for_phase("Phase2-Thermodynamic");
        assert_eq!(actions.len(), 8);
    }
}
