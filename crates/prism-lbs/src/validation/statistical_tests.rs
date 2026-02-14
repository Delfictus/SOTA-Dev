//! Statistical significance tests for method comparison
//!
//! Implements:
//! - McNemar's test for paired comparisons
//! - Bootstrap confidence intervals
//! - Wilcoxon signed-rank test

use rand::prelude::*;
use serde::{Deserialize, Serialize};

/// McNemar's test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McNemarResult {
    pub chi_squared: f64,
    pub p_value: f64,
    pub n_discordant: usize,
    pub a_better: usize,
    pub b_better: usize,
}

impl McNemarResult {
    /// Check if difference is statistically significant
    pub fn is_significant(&self, alpha: f64) -> bool {
        self.p_value < alpha
    }

    /// Human-readable interpretation
    pub fn interpret(&self) -> String {
        if self.p_value < 0.001 {
            format!(
                "Highly significant difference (p < 0.001, χ² = {:.2})",
                self.chi_squared
            )
        } else if self.p_value < 0.01 {
            format!(
                "Very significant difference (p < 0.01, χ² = {:.2})",
                self.chi_squared
            )
        } else if self.p_value < 0.05 {
            format!(
                "Significant difference (p < 0.05, χ² = {:.2})",
                self.chi_squared
            )
        } else {
            format!(
                "No significant difference (p = {:.3}, χ² = {:.2})",
                self.p_value, self.chi_squared
            )
        }
    }
}

/// McNemar's test for comparing two methods on the same dataset
///
/// Tests whether two methods have significantly different success rates
/// on paired samples.
///
/// # Arguments
/// * `method_a_correct` - Boolean array indicating if method A was correct
/// * `method_b_correct` - Boolean array indicating if method B was correct
///
/// # Returns
/// McNemar's test result with chi-squared statistic and p-value
pub fn mcnemar_test(method_a_correct: &[bool], method_b_correct: &[bool]) -> McNemarResult {
    assert_eq!(
        method_a_correct.len(),
        method_b_correct.len(),
        "Arrays must have same length"
    );

    // Count discordant pairs
    let mut b = 0; // A correct, B wrong
    let mut c = 0; // A wrong, B correct

    for (a, b_val) in method_a_correct.iter().zip(method_b_correct.iter()) {
        match (*a, *b_val) {
            (true, false) => b += 1,
            (false, true) => c += 1,
            _ => {}
        }
    }

    let b_f = b as f64;
    let c_f = c as f64;

    // McNemar's chi-squared statistic with continuity correction
    let chi_squared = if b_f + c_f > 0.0 {
        ((b_f - c_f).abs() - 1.0).max(0.0).powi(2) / (b_f + c_f)
    } else {
        0.0
    };

    // p-value from chi-squared distribution (1 df)
    let p_value = chi_squared_p_value(chi_squared, 1);

    McNemarResult {
        chi_squared,
        p_value,
        n_discordant: b + c,
        a_better: b,
        b_better: c,
    }
}

/// Confidence interval
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceInterval {
    pub mean: f64,
    pub lower: f64,
    pub upper: f64,
    pub std_error: f64,
}

impl ConfidenceInterval {
    /// Format as "mean [lower, upper]"
    pub fn format(&self, precision: usize) -> String {
        format!(
            "{:.prec$} [{:.prec$}, {:.prec$}]",
            self.mean,
            self.lower,
            self.upper,
            prec = precision
        )
    }

    /// Format as percentage
    pub fn format_percent(&self, precision: usize) -> String {
        format!(
            "{:.prec$}% [{:.prec$}%, {:.prec$}%]",
            self.mean * 100.0,
            self.lower * 100.0,
            self.upper * 100.0,
            prec = precision
        )
    }
}

/// Bootstrap confidence interval for a metric
///
/// # Arguments
/// * `values` - Sample values
/// * `n_bootstrap` - Number of bootstrap iterations (typically 10000)
/// * `confidence` - Confidence level (e.g., 0.95 for 95% CI)
/// * `seed` - Random seed for reproducibility
pub fn bootstrap_ci(
    values: &[f64],
    n_bootstrap: usize,
    confidence: f64,
    seed: u64,
) -> ConfidenceInterval {
    if values.is_empty() {
        return ConfidenceInterval {
            mean: 0.0,
            lower: 0.0,
            upper: 0.0,
            std_error: 0.0,
        };
    }

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let n = values.len();

    let mut bootstrap_means = Vec::with_capacity(n_bootstrap);

    for _ in 0..n_bootstrap {
        // Resample with replacement
        let sample: Vec<f64> = (0..n).map(|_| values[rng.gen_range(0..n)]).collect();
        let mean: f64 = sample.iter().sum::<f64>() / n as f64;
        bootstrap_means.push(mean);
    }

    bootstrap_means.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let alpha = 1.0 - confidence;
    let lower_idx = ((alpha / 2.0) * n_bootstrap as f64) as usize;
    let upper_idx = ((1.0 - alpha / 2.0) * n_bootstrap as f64) as usize;

    let mean: f64 = values.iter().sum::<f64>() / n as f64;
    let variance: f64 = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    let std_error = (variance / n as f64).sqrt();

    ConfidenceInterval {
        mean,
        lower: bootstrap_means[lower_idx.min(n_bootstrap - 1)],
        upper: bootstrap_means[upper_idx.min(n_bootstrap - 1)],
        std_error,
    }
}

/// Bootstrap CI for a success rate (binary outcomes)
pub fn bootstrap_ci_binary(
    successes: &[bool],
    n_bootstrap: usize,
    confidence: f64,
    seed: u64,
) -> ConfidenceInterval {
    let values: Vec<f64> = successes.iter().map(|&s| if s { 1.0 } else { 0.0 }).collect();
    bootstrap_ci(&values, n_bootstrap, confidence, seed)
}

/// Wilcoxon signed-rank test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WilcoxonResult {
    pub w_statistic: f64,
    pub p_value: f64,
    pub n_nonzero: usize,
}

impl WilcoxonResult {
    pub fn is_significant(&self, alpha: f64) -> bool {
        self.p_value < alpha
    }
}

/// Wilcoxon signed-rank test for paired samples
///
/// Non-parametric test for comparing two related samples.
///
/// # Arguments
/// * `differences` - Paired differences (method A - method B)
pub fn wilcoxon_signed_rank(differences: &[f64]) -> WilcoxonResult {
    // Remove zeros
    let nonzero: Vec<(f64, f64)> = differences
        .iter()
        .filter(|&&d| d != 0.0)
        .map(|&d| (d.abs(), d.signum()))
        .collect();

    let n = nonzero.len();
    if n == 0 {
        return WilcoxonResult {
            w_statistic: 0.0,
            p_value: 1.0,
            n_nonzero: 0,
        };
    }

    // Rank by absolute value
    let mut ranked: Vec<(usize, f64, f64)> = nonzero
        .iter()
        .enumerate()
        .map(|(i, (abs, sign))| (i, *abs, *sign))
        .collect();
    ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    // Assign ranks (handling ties with average rank)
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && (ranked[j].1 - ranked[i].1).abs() < 1e-10 {
            j += 1;
        }
        let avg_rank = (i + 1 + j) as f64 / 2.0;
        for k in i..j {
            ranks[ranked[k].0] = avg_rank;
        }
        i = j;
    }

    // Calculate W+ (sum of positive ranks)
    let w_plus: f64 = nonzero
        .iter()
        .enumerate()
        .filter(|(_, (_, sign))| *sign > 0.0)
        .map(|(i, _)| ranks[i])
        .sum();

    // Expected value and variance under null
    let expected = n as f64 * (n as f64 + 1.0) / 4.0;
    let variance = n as f64 * (n as f64 + 1.0) * (2.0 * n as f64 + 1.0) / 24.0;

    // Z-score (normal approximation)
    let z = if variance > 0.0 {
        (w_plus - expected) / variance.sqrt()
    } else {
        0.0
    };

    // Two-tailed p-value
    let p_value = 2.0 * (1.0 - normal_cdf(z.abs()));

    WilcoxonResult {
        w_statistic: w_plus,
        p_value: p_value.clamp(0.0, 1.0),
        n_nonzero: n,
    }
}

/// Paired t-test for comparing two methods
///
/// # Arguments
/// * `differences` - Paired differences (method A - method B)
///
/// # Returns
/// (t-statistic, p-value, degrees of freedom)
pub fn paired_t_test(differences: &[f64]) -> (f64, f64, usize) {
    let n = differences.len();
    if n < 2 {
        return (0.0, 1.0, 0);
    }

    let mean: f64 = differences.iter().sum::<f64>() / n as f64;
    let variance: f64 = differences.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    let std_error = (variance / n as f64).sqrt();

    let t_stat = if std_error > 0.0 {
        mean / std_error
    } else {
        0.0
    };

    let df = n - 1;
    let p_value = 2.0 * (1.0 - t_distribution_cdf(t_stat.abs(), df));

    (t_stat, p_value.clamp(0.0, 1.0), df)
}

// Statistical helper functions

fn chi_squared_p_value(chi_sq: f64, df: usize) -> f64 {
    // Use incomplete gamma function approximation
    let k = df as f64 / 2.0;
    let x = chi_sq / 2.0;
    1.0 - incomplete_gamma(k, x)
}

fn incomplete_gamma(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }

    // Series approximation for regularized incomplete gamma
    let mut sum = 0.0;
    let mut term = 1.0 / a;
    sum += term;

    for n in 1..200 {
        term *= x / (a + n as f64);
        sum += term;
        if term.abs() < 1e-12 {
            break;
        }
    }

    let gamma_a = gamma(a);
    if gamma_a == 0.0 {
        return 0.0;
    }

    (sum * x.powf(a) * (-x).exp() / gamma_a).clamp(0.0, 1.0)
}

fn gamma(x: f64) -> f64 {
    // Lanczos approximation
    if x < 0.5 {
        std::f64::consts::PI / ((std::f64::consts::PI * x).sin() * gamma(1.0 - x))
    } else {
        let x = x - 1.0;
        let g = 7_usize;
        let c = [
            0.99999999999980993,
            676.5203681218851,
            -1259.1392167224028,
            771.32342877765313,
            -176.61502916214059,
            12.507343278686905,
            -0.13857109526572012,
            9.9843695780195716e-6,
            1.5056327351493116e-7,
        ];

        let t = x + g as f64 + 0.5;
        let mut a = c[0];

        for i in 1..9 {
            a += c[i] / (x + i as f64);
        }

        (2.0 * std::f64::consts::PI).sqrt() * t.powf(x + 0.5) * (-t).exp() * a
    }
}

fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

fn erf(x: f64) -> f64 {
    // Approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

fn t_distribution_cdf(t: f64, df: usize) -> f64 {
    // Approximation using normal for large df
    if df > 100 {
        return normal_cdf(t);
    }

    // Use regularized incomplete beta function
    let x = df as f64 / (df as f64 + t * t);
    let ibeta = incomplete_beta(df as f64 / 2.0, 0.5, x);

    if t >= 0.0 {
        1.0 - ibeta / 2.0
    } else {
        ibeta / 2.0
    }
}

fn incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    // Simple approximation
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    // Continued fraction approximation (simplified)
    let bt = if x == 0.0 || x == 1.0 {
        0.0
    } else {
        (gamma(a + b) / (gamma(a) * gamma(b))) * x.powf(a) * (1.0 - x).powf(b)
    };

    // Use symmetry if needed
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * betacf(a, b, x) / a
    } else {
        1.0 - bt * betacf(b, a, 1.0 - x) / b
    }
}

fn betacf(a: f64, b: f64, x: f64) -> f64 {
    // Continued fraction for incomplete beta
    let max_iter = 100;
    let eps = 1e-10;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;

    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;

    if d.abs() < 1e-30 {
        d = 1e-30;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=max_iter {
        let m = m as f64;
        let m2 = 2.0 * m;

        // Even step
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + aa / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        h *= d * c;

        // Odd step
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + aa / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;

        if (del - 1.0).abs() < eps {
            break;
        }
    }

    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mcnemar() {
        // Method A gets 8 right, B gets 5 right
        // 3 cases: A right, B wrong
        // 0 cases: A wrong, B right
        let a = vec![true, true, true, true, true, true, true, true, false, false];
        let b = vec![true, true, true, true, true, false, false, false, false, false];

        let result = mcnemar_test(&a, &b);
        assert_eq!(result.a_better, 3);
        assert_eq!(result.b_better, 0);
    }

    #[test]
    fn test_bootstrap_ci() {
        let values: Vec<f64> = (0..100).map(|x| x as f64).collect();
        let ci = bootstrap_ci(&values, 1000, 0.95, 42);

        assert!((ci.mean - 49.5).abs() < 1.0);
        assert!(ci.lower < ci.mean);
        assert!(ci.upper > ci.mean);
    }

    #[test]
    fn test_wilcoxon() {
        // Positive differences (method A better)
        let diff = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = wilcoxon_signed_rank(&diff);
        assert!(result.p_value < 0.1);
    }

    #[test]
    fn test_paired_t() {
        let diff = vec![1.0, 1.5, 2.0, 1.2, 1.8];
        let (t, p, df) = paired_t_test(&diff);
        assert!(t > 0.0);
        assert!(p < 0.05);
        assert_eq!(df, 4);
    }
}
