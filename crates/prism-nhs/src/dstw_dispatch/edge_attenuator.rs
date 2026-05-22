use core::fmt;

use super::types::{
    ChannelCapacity, ComplementPenalty, DTSGEdge, FrustrationPenalty, HysteresisCapacity,
    HysteresisEnthalpy, PoseUncertainty, ScalingConstant, TransferEntropy,
};

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct ScalingConstants {
    pub beta_frustration: ScalingConstant,
    pub beta_stabilization: ScalingConstant,
    pub te_channel_capacity: ChannelCapacity,
    pub hysteresis_capacity: HysteresisCapacity,
}

impl ScalingConstants {
    #[must_use]
    pub const fn new(
        beta_frustration: ScalingConstant,
        beta_stabilization: ScalingConstant,
        te_channel_capacity: ChannelCapacity,
        hysteresis_capacity: HysteresisCapacity,
    ) -> Self {
        Self {
            beta_frustration,
            beta_stabilization,
            te_channel_capacity,
            hysteresis_capacity,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct EdgePenalty {
    pub clash: FrustrationPenalty,
    pub complement: ComplementPenalty,
}

impl EdgePenalty {
    #[must_use]
    pub const fn new(clash: FrustrationPenalty, complement: ComplementPenalty) -> Self {
        Self { clash, complement }
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct PerturbedEdgeStats {
    pub edge: DTSGEdge,
    pub te_out_mean: TransferEntropy,
    pub te_in_mean: TransferEntropy,
    pub delta_hc_mean: HysteresisEnthalpy,
    pub u_pose_te: PoseUncertainty,
    pub u_pose_hc: PoseUncertainty,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AttenuationError {
    EmptyConformerSet,
    EdgeCountMismatch {
        conformer_index: usize,
        expected: usize,
        actual: usize,
    },
    NonFiniteScale {
        edge_index: usize,
        conformer_index: usize,
        value: f64,
    },
    InvalidQuantity(String),
}

impl fmt::Display for AttenuationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyConformerSet => write!(formatter, "at least one conformer is required"),
            Self::EdgeCountMismatch {
                conformer_index,
                expected,
                actual,
            } => write!(
                formatter,
                "conformer {conformer_index} has {actual} penalties, expected {expected}"
            ),
            Self::NonFiniteScale {
                edge_index,
                conformer_index,
                value,
            } => write!(
                formatter,
                "non-finite attenuation scale for conformer {conformer_index}, edge {edge_index}: {value}"
            ),
            Self::InvalidQuantity(message) => write!(formatter, "{message}"),
        }
    }
}

impl std::error::Error for AttenuationError {}

#[derive(Debug, Copy, Clone)]
struct RunningVariance {
    count: u32,
    mean: f64,
    m2: f64,
}

impl RunningVariance {
    const fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
        }
    }

    fn push(&mut self, value: f64) {
        self.count += 1;
        let count = f64::from(self.count);
        let delta = value - self.mean;
        self.mean += delta / count;
        let delta_after = value - self.mean;
        self.m2 += delta * delta_after;
    }

    #[must_use]
    const fn mean(self) -> f64 {
        self.mean
    }

    #[must_use]
    fn population_variance(self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.m2 / f64::from(self.count)
        }
    }
}

pub fn attenuate_edges(
    edges: &[DTSGEdge],
    conformer_penalties: &[Vec<EdgePenalty>],
    scaling: ScalingConstants,
) -> Result<Vec<PerturbedEdgeStats>, AttenuationError> {
    if conformer_penalties.is_empty() {
        return Err(AttenuationError::EmptyConformerSet);
    }
    for (conformer_index, penalties) in conformer_penalties.iter().enumerate() {
        if penalties.len() != edges.len() {
            return Err(AttenuationError::EdgeCountMismatch {
                conformer_index,
                expected: edges.len(),
                actual: penalties.len(),
            });
        }
    }

    let mut output = Vec::with_capacity(edges.len());
    for (edge_index, edge) in edges.iter().copied().enumerate() {
        let mut te_out = RunningVariance::new();
        let mut te_in = RunningVariance::new();
        let mut delta_hc = RunningVariance::new();

        for (conformer_index, penalties) in conformer_penalties.iter().enumerate() {
            let penalty = penalties[edge_index];
            let perturbed = perturb_edge_metrics(&edge, penalty, scaling).map_err(|value| {
                AttenuationError::NonFiniteScale {
                    edge_index,
                    conformer_index,
                    value,
                }
            })?;
            te_out.push(perturbed.te_out);
            te_in.push(perturbed.te_in);
            delta_hc.push(perturbed.delta_hc);
        }

        output.push(PerturbedEdgeStats {
            edge,
            te_out_mean: TransferEntropy::new(te_out.mean())
                .map_err(|err| AttenuationError::InvalidQuantity(err.to_string()))?,
            te_in_mean: TransferEntropy::new(te_in.mean())
                .map_err(|err| AttenuationError::InvalidQuantity(err.to_string()))?,
            delta_hc_mean: HysteresisEnthalpy::new(delta_hc.mean())
                .map_err(|err| AttenuationError::InvalidQuantity(err.to_string()))?,
            u_pose_te: PoseUncertainty::new(te_out.population_variance())
                .map_err(|err| AttenuationError::InvalidQuantity(err.to_string()))?,
            u_pose_hc: PoseUncertainty::new(delta_hc.population_variance())
                .map_err(|err| AttenuationError::InvalidQuantity(err.to_string()))?,
        });
    }
    Ok(output)
}

#[derive(Debug, Copy, Clone)]
struct PerturbedMetricValues {
    te_out: f64,
    te_in: f64,
    delta_hc: f64,
}

fn perturb_edge_metrics(
    edge: &DTSGEdge,
    penalty: EdgePenalty,
    scaling: ScalingConstants,
) -> Result<PerturbedMetricValues, f64> {
    let destructive_scale = destructive_interference_scale(penalty, scaling);
    let constructive_drive = constructive_interference_drive(penalty, scaling);
    let te_out = saturated_transfer_entropy(
        edge.metrics.te_out.get() * destructive_scale,
        constructive_drive,
        scaling.te_channel_capacity,
    );
    let te_in = saturated_transfer_entropy(
        edge.metrics.te_in.get() * destructive_scale,
        constructive_drive,
        scaling.te_channel_capacity,
    );
    let delta_hc = saturated_hysteresis_enthalpy(
        edge.metrics.delta_hc.get() * destructive_scale,
        constructive_drive,
        scaling.hysteresis_capacity,
    );
    for value in [te_out, te_in, delta_hc] {
        if !value.is_finite() {
            return Err(value);
        }
    }
    Ok(PerturbedMetricValues {
        te_out,
        te_in,
        delta_hc,
    })
}

fn destructive_interference_scale(penalty: EdgePenalty, scaling: ScalingConstants) -> f64 {
    let argument = -scaling.beta_frustration.get() * penalty.clash.get();
    argument.exp()
}

fn constructive_interference_drive(penalty: EdgePenalty, scaling: ScalingConstants) -> f64 {
    scaling.beta_stabilization.get() * penalty.complement.get()
}

fn saturated_transfer_entropy(
    base_after_clash: f64,
    constructive_drive: f64,
    capacity: ChannelCapacity,
) -> f64 {
    saturated_signed_magnitude(base_after_clash, constructive_drive, capacity.get())
}

fn saturated_hysteresis_enthalpy(
    base_after_clash: f64,
    constructive_drive: f64,
    capacity: HysteresisCapacity,
) -> f64 {
    saturated_signed_magnitude(base_after_clash, constructive_drive, capacity.get())
}

fn saturated_signed_magnitude(
    base_after_clash: f64,
    constructive_drive: f64,
    channel_capacity: f64,
) -> f64 {
    if base_after_clash == 0.0 {
        return 0.0;
    }
    let sign = base_after_clash.signum();
    let magnitude = base_after_clash.abs();
    if magnitude >= channel_capacity {
        return sign * channel_capacity;
    }
    let bounded_drive = if constructive_drive.is_finite() {
        constructive_drive.max(0.0)
    } else {
        f64::INFINITY
    };
    let log_ratio = (channel_capacity - magnitude).ln() - magnitude.ln();
    let logistic_argument = log_ratio - bounded_drive;
    let saturated_magnitude = if logistic_argument >= 0.0 {
        let exp_negative = (-logistic_argument).exp();
        channel_capacity * exp_negative / (1.0 + exp_negative)
    } else {
        channel_capacity / (1.0 + logistic_argument.exp())
    };
    sign * saturated_magnitude
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dstw_dispatch::{
        ChannelCapacity, DTSGEdgeMetrics, HydrationVariance, HysteresisCapacity,
        HysteresisEnthalpy, HysteresisPersistence, ResidueIdx, SpatialVariance,
    };

    fn edge() -> DTSGEdge {
        DTSGEdge::new(
            ResidueIdx::new(144),
            ResidueIdx::new(145),
            DTSGEdgeMetrics::new(
                TransferEntropy::new(2.0).unwrap_or_else(|err| panic!("{err}")),
                TransferEntropy::new(1.0).unwrap_or_else(|err| panic!("{err}")),
                HysteresisEnthalpy::new(4.0).unwrap_or_else(|err| panic!("{err}")),
                HydrationVariance::new(0.2).unwrap_or_else(|err| panic!("{err}")),
                SpatialVariance::new(0.3).unwrap_or_else(|err| panic!("{err}")),
                HysteresisPersistence::new(0.8).unwrap_or_else(|err| panic!("{err}")),
            )
            .unwrap_or_else(|err| panic!("{err}")),
        )
    }

    #[test]
    fn bivariate_interference_supports_attenuation_and_amplification() {
        let scaling = ScalingConstants::new(
            ScalingConstant::new(1.0).unwrap_or_else(|err| panic!("{err}")),
            ScalingConstant::new(0.6).unwrap_or_else(|err| panic!("{err}")),
            ChannelCapacity::new(10.0).unwrap_or_else(|err| panic!("{err}")),
            HysteresisCapacity::new(20.0).unwrap_or_else(|err| panic!("{err}")),
        );
        let penalties = vec![
            vec![EdgePenalty::new(
                FrustrationPenalty::new(1.0).unwrap_or_else(|err| panic!("{err}")),
                ComplementPenalty::new(0.0).unwrap_or_else(|err| panic!("{err}")),
            )],
            vec![EdgePenalty::new(
                FrustrationPenalty::new(0.0).unwrap_or_else(|err| panic!("{err}")),
                ComplementPenalty::new(1.0).unwrap_or_else(|err| panic!("{err}")),
            )],
        ];
        let stats =
            attenuate_edges(&[edge()], &penalties, scaling).unwrap_or_else(|err| panic!("{err}"));
        let mean = stats[0].te_out_mean.get();
        let attenuated = 2.0_f64 * (-1.0_f64).exp();
        let amplified = saturated_transfer_entropy(
            2.0,
            0.6,
            ChannelCapacity::new(10.0).unwrap_or_else(|err| panic!("{err}")),
        );
        let expected = (attenuated + amplified) / 2.0;
        assert!((mean - expected).abs() < 1.0e-12);
        assert!(stats[0].u_pose_te.get() > 0.0);
    }

    #[test]
    fn rejects_missing_conformers() {
        let scaling = ScalingConstants::new(
            ScalingConstant::new(1.0).unwrap_or_else(|err| panic!("{err}")),
            ScalingConstant::new(0.6).unwrap_or_else(|err| panic!("{err}")),
            ChannelCapacity::new(10.0).unwrap_or_else(|err| panic!("{err}")),
            HysteresisCapacity::new(20.0).unwrap_or_else(|err| panic!("{err}")),
        );
        let result = attenuate_edges(&[edge()], &[], scaling);
        assert!(matches!(result, Err(AttenuationError::EmptyConformerSet)));
    }

    #[test]
    fn rejects_edge_count_mismatch() {
        let scaling = ScalingConstants::new(
            ScalingConstant::new(1.0).unwrap_or_else(|err| panic!("{err}")),
            ScalingConstant::new(0.6).unwrap_or_else(|err| panic!("{err}")),
            ChannelCapacity::new(10.0).unwrap_or_else(|err| panic!("{err}")),
            HysteresisCapacity::new(20.0).unwrap_or_else(|err| panic!("{err}")),
        );
        let result = attenuate_edges(&[edge()], &[Vec::new()], scaling);
        assert!(matches!(
            result,
            Err(AttenuationError::EdgeCountMismatch { .. })
        ));
    }

    #[test]
    fn rejects_invalid_hysteresis_persistence_before_edge_construction() {
        assert!(HysteresisPersistence::new(f64::NAN).is_err());
        assert!(HysteresisPersistence::new(1.1).is_err());
        assert!(HysteresisPersistence::new(0.0).is_ok());
        assert!(HysteresisPersistence::new(1.0).is_ok());
    }

    #[test]
    fn rejects_negative_nonnegative_domain_quantities() {
        assert!(HydrationVariance::new(-1.0).is_err());
        assert!(PoseUncertainty::new(-1.0).is_err());
        assert!(ScalingConstant::new(-1.0).is_err());
        assert!(HysteresisCapacity::new(-1.0).is_err());
    }

    #[test]
    fn constructive_interference_saturates_at_channel_capacity() {
        let capacity = ChannelCapacity::new(3.0).unwrap_or_else(|err| panic!("{err}"));
        let saturated = saturated_transfer_entropy(2.0, 1.0e300, capacity);
        assert!(saturated <= capacity.get());
        assert!((saturated - capacity.get()).abs() < 1.0e-12);
    }
}
