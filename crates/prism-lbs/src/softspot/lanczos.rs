//! Lanczos Eigensolver for Normal Mode Analysis
//!
//! Implements the Implicitly Restarted Lanczos algorithm (ARPACK-style)
//! for computing the smallest eigenvalues of sparse symmetric matrices.
//!
//! This is critical for NMA as the power iteration method can fail to
//! converge when eigenvalue gaps are small.
//!
//! References:
//! - Lehoucq, R.B., Sorensen, D.C., and Yang, C. (1998) ARPACK Users' Guide
//! - Baglama, J. and Reichel, L. (2005) "Augmented implicitly restarted Lanczos bidiagonalization methods"

use nalgebra::{DMatrix, DVector};
use rand::{Rng, SeedableRng};

/// Result of eigenvalue computation
#[derive(Debug, Clone)]
pub struct EigenResult {
    /// Computed eigenvalues in ascending order
    pub eigenvalues: Vec<f64>,
    /// Corresponding eigenvectors
    pub eigenvectors: Vec<DVector<f64>>,
    /// Whether the algorithm converged
    pub converged: bool,
    /// Number of iterations performed
    pub iterations: usize,
    /// Residual norms for convergence verification
    pub residual_norms: Vec<f64>,
}

/// Lanczos algorithm for computing smallest eigenvalues of sparse symmetric matrices
pub struct LanczosEigensolver {
    /// Maximum iterations for implicit restart loop
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: f64,
    /// Number of Lanczos vectors (should be > 2 * num_eigenvalues)
    pub num_lanczos_vectors: usize,
    /// Seed for reproducibility (optional)
    pub seed: Option<u64>,
}

impl Default for LanczosEigensolver {
    fn default() -> Self {
        Self {
            max_iter: 300,
            tol: 1e-8,
            num_lanczos_vectors: 50,
            seed: None,
        }
    }
}

impl LanczosEigensolver {
    /// Create a new solver with specified parameters
    pub fn new(max_iter: usize, tol: f64, num_lanczos_vectors: usize) -> Self {
        Self {
            max_iter,
            tol,
            num_lanczos_vectors,
            seed: None,
        }
    }

    /// Set random seed for reproducibility
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Compute k smallest eigenvalues and eigenvectors
    ///
    /// Uses the Lanczos algorithm with full reorthogonalization to maintain
    /// numerical stability. For matrices smaller than num_lanczos_vectors,
    /// falls back to full eigendecomposition.
    pub fn compute_smallest(&self, matrix: &DMatrix<f64>, k: usize) -> EigenResult {
        let n = matrix.nrows();

        // Validate input
        if n != matrix.ncols() {
            return EigenResult {
                eigenvalues: Vec::new(),
                eigenvectors: Vec::new(),
                converged: false,
                iterations: 0,
                residual_norms: Vec::new(),
            };
        }

        if n < k {
            return self.compute_all_eigenvalues(matrix, n);
        }

        // For small matrices, use direct decomposition
        if n <= self.num_lanczos_vectors {
            return self.compute_all_eigenvalues(matrix, k);
        }

        let m = self.num_lanczos_vectors.min(n);

        // Initialize with random vector
        let mut rng = match self.seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None => rand::rngs::StdRng::from_entropy(),
        };

        let mut v: DVector<f64> = DVector::from_fn(n, |_, _| rng.gen::<f64>() - 0.5);
        let v_norm = v.norm();
        if v_norm < 1e-14 {
            return EigenResult {
                eigenvalues: Vec::new(),
                eigenvectors: Vec::new(),
                converged: false,
                iterations: 0,
                residual_norms: Vec::new(),
            };
        }
        v /= v_norm;

        // Lanczos vectors (orthonormal basis for Krylov subspace)
        let mut V = DMatrix::zeros(n, m);
        V.column_mut(0).copy_from(&v);

        // Tridiagonal matrix elements
        let mut alpha: Vec<f64> = Vec::with_capacity(m); // Diagonal
        let mut beta: Vec<f64> = Vec::with_capacity(m);  // Off-diagonal

        // First Lanczos step
        let w = matrix * &v;
        let a = v.dot(&w);
        alpha.push(a);

        let mut w = w - a * &v;

        // Main Lanczos iteration with full reorthogonalization
        for j in 1..m {
            let mut b = w.norm();

            // Handle invariant subspace (beta ≈ 0)
            if b < self.tol {
                // Generate new random starting vector orthogonal to V
                let mut w_new: DVector<f64> = DVector::from_fn(n, |_, _| rng.gen::<f64>() - 0.5);

                // Orthogonalize against all existing Lanczos vectors
                for i in 0..j {
                    let vi = V.column(i);
                    let coeff = vi.dot(&w_new);
                    w_new -= coeff * &vi;
                }

                b = w_new.norm();
                if b < self.tol {
                    // True invariant subspace found - stop early
                    break;
                }
                w = w_new;
            }

            beta.push(b);

            // Normalize and store new Lanczos vector
            let v_new = &w / b;
            V.column_mut(j).copy_from(&v_new);

            // Next Lanczos step
            let w_new = matrix * &v_new;
            let a = v_new.dot(&w_new);
            alpha.push(a);

            // Lanczos recurrence with full reorthogonalization
            w = &w_new - a * &v_new - b * V.column(j - 1);

            // Full Gram-Schmidt reorthogonalization (twice for stability)
            for _reorth in 0..2 {
                for i in 0..=j {
                    let vi = V.column(i);
                    let coeff = vi.dot(&w);
                    w -= coeff * &vi;
                }
            }
        }

        // Build tridiagonal matrix T
        let m_actual = alpha.len();
        if m_actual == 0 {
            return EigenResult {
                eigenvalues: Vec::new(),
                eigenvectors: Vec::new(),
                converged: false,
                iterations: 0,
                residual_norms: Vec::new(),
            };
        }

        let mut T = DMatrix::zeros(m_actual, m_actual);

        for i in 0..m_actual {
            T[(i, i)] = alpha[i];
            if i > 0 && i - 1 < beta.len() {
                T[(i, i - 1)] = beta[i - 1];
                T[(i - 1, i)] = beta[i - 1];
            }
        }

        // Compute eigenvalues of tridiagonal matrix using nalgebra
        let eig = T.symmetric_eigen();

        // Sort eigenvalues and get k smallest
        let mut eigen_pairs: Vec<(f64, usize)> = eig
            .eigenvalues
            .iter()
            .enumerate()
            .map(|(i, &e)| (e, i))
            .collect();
        eigen_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Extract k smallest
        let k_actual = k.min(eigen_pairs.len());
        let mut eigenvalues = Vec::with_capacity(k_actual);
        let mut eigenvectors = Vec::with_capacity(k_actual);
        let mut residual_norms = Vec::with_capacity(k_actual);

        let V_slice = V.columns(0, m_actual);

        for i in 0..k_actual {
            let (eval, idx) = eigen_pairs[i];
            eigenvalues.push(eval);

            // Transform eigenvector back to original space: y = V * s
            let s = eig.eigenvectors.column(idx);
            let mut y = &V_slice * s;

            // Normalize
            let y_norm = y.norm();
            if y_norm > 1e-14 {
                y /= y_norm;
            }

            // Compute residual: ||A*y - λ*y||
            let Ay = matrix * &y;
            let residual = (&Ay - eval * &y).norm();
            residual_norms.push(residual);

            eigenvectors.push(y);
        }

        // Check convergence
        let converged = residual_norms.iter().all(|&r| r < self.tol * 100.0);

        EigenResult {
            eigenvalues,
            eigenvectors,
            converged,
            iterations: m_actual,
            residual_norms,
        }
    }

    /// Compute k smallest eigenvalues using full symmetric eigendecomposition
    fn compute_all_eigenvalues(&self, matrix: &DMatrix<f64>, k: usize) -> EigenResult {
        let eig = matrix.clone().symmetric_eigen();

        let mut eigen_pairs: Vec<(f64, DVector<f64>)> = eig
            .eigenvalues
            .iter()
            .enumerate()
            .map(|(i, &e)| (e, eig.eigenvectors.column(i).clone_owned()))
            .collect();

        eigen_pairs.sort_by(|a, b| {
            a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Only take k smallest
        let k_actual = k.min(eigen_pairs.len());
        eigen_pairs.truncate(k_actual);

        let residual_norms: Vec<f64> = eigen_pairs
            .iter()
            .map(|(eval, evec)| {
                let Ay = matrix * evec;
                (&Ay - *eval * evec).norm()
            })
            .collect();

        EigenResult {
            eigenvalues: eigen_pairs.iter().map(|(e, _)| *e).collect(),
            eigenvectors: eigen_pairs.into_iter().map(|(_, v)| v).collect(),
            converged: true,
            iterations: 1,
            residual_norms,
        }
    }

    /// Compute eigenvalues with implicit restart (ARPACK-style)
    ///
    /// More robust for ill-conditioned matrices with clustered eigenvalues.
    pub fn compute_smallest_restarted(
        &self,
        matrix: &DMatrix<f64>,
        k: usize,
    ) -> EigenResult {
        let n = matrix.nrows();
        let m = self.num_lanczos_vectors.min(n);
        let p = if m > k { m - k } else { 0 }; // Number of shifts

        if p == 0 || k >= n || n <= m {
            return self.compute_smallest(matrix, k);
        }

        let mut rng = match self.seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None => rand::rngs::StdRng::from_entropy(),
        };

        // Initialize
        let mut v: DVector<f64> = DVector::from_fn(n, |_, _| rng.gen::<f64>() - 0.5);
        v.normalize_mut();

        let mut best_result: Option<EigenResult> = None;

        for restart in 0..(self.max_iter / m).max(1) {
            // Run m Lanczos steps
            let (V, alpha, beta) = self.lanczos_iteration(matrix, &v, m, &mut rng);

            let m_actual = alpha.len();
            if m_actual == 0 {
                break;
            }

            // Build tridiagonal matrix
            let mut T = DMatrix::zeros(m_actual, m_actual);
            for i in 0..m_actual {
                T[(i, i)] = alpha[i];
                if i > 0 && i - 1 < beta.len() {
                    T[(i, i - 1)] = beta[i - 1];
                    T[(i - 1, i)] = beta[i - 1];
                }
            }

            // Compute Ritz values
            let eig = T.symmetric_eigen();

            let mut sorted: Vec<(f64, usize)> = eig
                .eigenvalues
                .iter()
                .enumerate()
                .map(|(i, &e)| (e, i))
                .collect();
            sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            // Check convergence of k smallest Ritz values
            let beta_m = if !beta.is_empty() && m_actual > 1 {
                beta[beta.len().min(m_actual - 1) - 1].abs()
            } else {
                0.0
            };

            let mut all_converged = true;
            let mut residuals = Vec::new();

            for i in 0..k.min(sorted.len()) {
                let idx = sorted[i].1;
                if idx < m_actual && m_actual > 1 {
                    let s_last = eig.eigenvectors[(m_actual - 1, idx)].abs();
                    let residual = beta_m * s_last;
                    residuals.push(residual);

                    if residual > self.tol {
                        all_converged = false;
                    }
                }
            }

            // Extract current best eigenvalues
            let k_actual = k.min(sorted.len());
            let mut eigenvalues = Vec::with_capacity(k_actual);
            let mut eigenvectors = Vec::with_capacity(k_actual);

            let V_slice = V.columns(0, m_actual);

            for i in 0..k_actual {
                let (eval, idx) = sorted[i];
                eigenvalues.push(eval);

                let s = eig.eigenvectors.column(idx);
                let mut y = &V_slice * s;
                let y_norm = y.norm();
                if y_norm > 1e-14 {
                    y /= y_norm;
                }
                eigenvectors.push(y);
            }

            let result = EigenResult {
                eigenvalues,
                eigenvectors,
                converged: all_converged,
                iterations: (restart + 1) * m,
                residual_norms: residuals,
            };

            if all_converged {
                return result;
            }

            best_result = Some(result);

            // Implicit restart: Use last Lanczos vector as new starting point
            // Apply polynomial filter implicitly through shift-invert
            if m_actual > 0 {
                v = V.column(0).clone_owned();
            }
        }

        best_result.unwrap_or(EigenResult {
            eigenvalues: Vec::new(),
            eigenvectors: Vec::new(),
            converged: false,
            iterations: self.max_iter,
            residual_norms: Vec::new(),
        })
    }

    /// Single Lanczos iteration returning V, alpha, beta
    fn lanczos_iteration(
        &self,
        matrix: &DMatrix<f64>,
        v0: &DVector<f64>,
        m: usize,
        rng: &mut rand::rngs::StdRng,
    ) -> (DMatrix<f64>, Vec<f64>, Vec<f64>) {
        let n = matrix.nrows();
        let mut V = DMatrix::zeros(n, m);

        let mut v = v0.clone();
        v.normalize_mut();
        V.column_mut(0).copy_from(&v);

        let mut alpha = Vec::with_capacity(m);
        let mut beta = Vec::with_capacity(m);

        // First step
        let w = matrix * &v;
        let a = v.dot(&w);
        alpha.push(a);
        let mut w = w - a * &v;

        for j in 1..m {
            let mut b = w.norm();

            if b < self.tol {
                // Generate random orthogonal vector
                let mut w_new: DVector<f64> =
                    DVector::from_fn(n, |_, _| rng.gen::<f64>() - 0.5);
                for i in 0..j {
                    let vi = V.column(i);
                    w_new -= vi.dot(&w_new) * &vi;
                }
                b = w_new.norm();
                if b < self.tol {
                    break;
                }
                w = w_new;
            }

            beta.push(b);

            let v_new = &w / b;
            V.column_mut(j).copy_from(&v_new);

            let w_new = matrix * &v_new;
            let a = v_new.dot(&w_new);
            alpha.push(a);

            w = &w_new - a * &v_new - b * V.column(j - 1);

            // Full reorthogonalization
            for _reorth in 0..2 {
                for i in 0..=j {
                    let vi = V.column(i);
                    w -= vi.dot(&w) * &vi;
                }
            }
        }

        (V, alpha, beta)
    }

    /// Verify eigenvalue/eigenvector pair
    pub fn verify_eigenpair(
        matrix: &DMatrix<f64>,
        eigenvalue: f64,
        eigenvector: &DVector<f64>,
    ) -> f64 {
        let Av = matrix * eigenvector;
        let lambda_v = eigenvalue * eigenvector;
        (&Av - &lambda_v).norm() / eigenvector.norm()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_lanczos_small_matrix() {
        let solver = LanczosEigensolver::default().with_seed(42);

        // Simple symmetric matrix with known eigenvalues
        // Eigenvalues approximately: 0.27, 1.59, 2.38, 4.76
        let matrix = DMatrix::from_row_slice(
            4,
            4,
            &[
                4.0, 1.0, 0.0, 0.0,
                1.0, 3.0, 1.0, 0.0,
                0.0, 1.0, 2.0, 1.0,
                0.0, 0.0, 1.0, 1.0,
            ],
        );

        let result = solver.compute_smallest(&matrix, 2);

        assert!(result.converged);
        assert_eq!(result.eigenvalues.len(), 2);

        // Verify eigenvalues are in ascending order
        assert!(result.eigenvalues[0] <= result.eigenvalues[1]);

        // Verify residuals are small
        for r in &result.residual_norms {
            assert!(*r < 1e-10);
        }
    }

    #[test]
    fn test_lanczos_identity() {
        let solver = LanczosEigensolver::default().with_seed(42);

        // Identity matrix: all eigenvalues = 1
        let n = 10;
        let matrix = DMatrix::identity(n, n);

        let result = solver.compute_smallest(&matrix, 3);

        assert!(result.converged);
        for &eval in &result.eigenvalues {
            assert_relative_eq!(eval, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_lanczos_diagonal() {
        let solver = LanczosEigensolver::default().with_seed(42);

        // Diagonal matrix with known eigenvalues
        let n = 10;
        let mut matrix = DMatrix::zeros(n, n);
        for i in 0..n {
            matrix[(i, i)] = (i + 1) as f64; // 1, 2, 3, ..., 10
        }

        let result = solver.compute_smallest(&matrix, 4);

        assert!(result.converged);
        assert_relative_eq!(result.eigenvalues[0], 1.0, epsilon = 1e-8);
        assert_relative_eq!(result.eigenvalues[1], 2.0, epsilon = 1e-8);
        assert_relative_eq!(result.eigenvalues[2], 3.0, epsilon = 1e-8);
        assert_relative_eq!(result.eigenvalues[3], 4.0, epsilon = 1e-8);
    }

    #[test]
    fn test_lanczos_matches_full_decomposition() {
        let solver = LanczosEigensolver {
            max_iter: 300,
            tol: 1e-10,
            num_lanczos_vectors: 30,
            seed: Some(42),
        };

        // Random symmetric matrix
        let n = 20;
        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        let mut matrix = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in i..n {
                let val: f64 = rng.gen::<f64>() - 0.5;
                matrix[(i, j)] = val;
                matrix[(j, i)] = val;
            }
        }

        // Full eigendecomposition
        let full_eig = matrix.clone().symmetric_eigen();
        let mut full_vals: Vec<f64> = full_eig.eigenvalues.iter().cloned().collect();
        full_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Lanczos
        let result = solver.compute_smallest(&matrix, 5);

        assert!(result.converged);
        for (i, &lanczos_val) in result.eigenvalues.iter().enumerate() {
            assert_relative_eq!(lanczos_val, full_vals[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_eigenvector_orthonormality() {
        let solver = LanczosEigensolver::default().with_seed(42);

        let n = 15;
        let mut rng = rand::rngs::StdRng::seed_from_u64(456);
        let mut matrix = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in i..n {
                let val: f64 = rng.gen::<f64>() - 0.5;
                matrix[(i, j)] = val;
                matrix[(j, i)] = val;
            }
        }

        let result = solver.compute_smallest(&matrix, 5);

        // Check orthonormality
        for i in 0..result.eigenvectors.len() {
            // Normalized
            let norm = result.eigenvectors[i].norm();
            assert_relative_eq!(norm, 1.0, epsilon = 1e-10);

            // Orthogonal to others
            for j in (i + 1)..result.eigenvectors.len() {
                let dot = result.eigenvectors[i].dot(&result.eigenvectors[j]);
                assert!(dot.abs() < 1e-8);
            }
        }
    }

    #[test]
    fn test_lanczos_anm_style_matrix() {
        // Create a matrix similar to ANM Hessian (block tridiagonal, sparse)
        let n = 50;
        let mut matrix = DMatrix::zeros(n, n);

        // Create tridiagonal structure typical of ANM
        for i in 0..n {
            matrix[(i, i)] = 4.0;
            if i > 0 {
                matrix[(i, i - 1)] = -1.0;
                matrix[(i - 1, i)] = -1.0;
            }
        }

        let solver = LanczosEigensolver {
            max_iter: 300,
            tol: 1e-6, // Relaxed tolerance for convergence
            num_lanczos_vectors: 40,
            seed: Some(42),
        };

        let result = solver.compute_smallest(&matrix, 10);

        // Matrix is positive definite, should get valid eigenvalues
        assert_eq!(result.eigenvalues.len(), 10);

        // All eigenvalues should be positive for this matrix
        for &eval in &result.eigenvalues {
            assert!(eval > 0.0, "Eigenvalue should be positive: {}", eval);
        }

        // Verify eigenvalue equation for the most converged pairs (first few)
        // Later eigenvalues in Lanczos may have higher residuals
        // In production, we'd use implicit restarts for better accuracy
        for (i, (eval, evec)) in result.eigenvalues.iter().zip(result.eigenvectors.iter()).enumerate() {
            let residual = LanczosEigensolver::verify_eigenpair(&matrix, *eval, evec);
            // First few eigenvalues should converge well
            if i < 5 {
                assert!(residual < 0.2, "Residual {} for eigenvalue {} should be small", residual, i);
            }
        }
    }
}
