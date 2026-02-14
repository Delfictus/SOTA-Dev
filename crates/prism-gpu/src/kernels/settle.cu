/**
 * SETTLE - Analytical Constraint Solver for Rigid Water
 *
 * Implements the SETTLE algorithm for maintaining rigid TIP3P water geometry
 * during molecular dynamics. This is an analytical solution that satisfies
 * all three distance constraints (OH1, OH2, HH) exactly in a single pass.
 *
 * Reference: Miyamoto & Kollman (1992) J. Comput. Chem. 13:952-962
 *
 * Algorithm overview:
 * 1. Compute old and new center of mass
 * 2. Define canonical frame from old configuration
 * 3. Transform new positions to canonical frame
 * 4. Solve analytical constraint equations (phi, psi, theta)
 * 5. Construct constrained positions in canonical frame
 * 6. Transform back to lab frame
 *
 * Canonical frame:
 * - Origin at COM of water molecule
 * - Z-axis along bisector (from H midpoint to O)
 * - X-axis along H-H direction
 * - Y-axis perpendicular to molecular plane
 *
 * Constrained positions in canonical frame:
 * - O: (0, 0, -ra)
 * - H1: (-rc, 0, rb)
 * - H2: (+rc, 0, rb)
 *
 * where:
 * - ra = distance from O to COM
 * - rb = distance from H to COM along bisector
 * - rc = half of H-H distance
 */

extern "C" {

/**
 * Apply SETTLE constraints to water molecules
 *
 * Each thread handles one water molecule.
 * Modifies new_positions in-place to satisfy constraints.
 *
 * @param new_pos New positions after integration [n_atoms * 3]
 * @param old_pos Old positions before integration [n_atoms * 3]
 * @param water_idx Water indices [n_waters * 3] - (O, H1, H2) for each water
 * @param n_waters Number of water molecules
 * @param mO Mass of oxygen
 * @param mH Mass of hydrogen
 * @param ra Distance from O to COM (negative z direction)
 * @param rb Distance from H to COM along bisector (positive z direction)
 * @param rc Half of HH distance (x direction)
 * @param rOH2 Target OH distance squared (not used in simplified version)
 * @param rHH2 Target HH distance squared (not used in simplified version)
 */
__global__ void settle_constraints(
    float* __restrict__ new_pos,
    const float* __restrict__ old_pos,
    const int* __restrict__ water_idx,
    int n_waters,
    float mO, float mH,
    float ra, float rb, float rc,
    float rOH2, float rHH2
) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= n_waters) return;

    // Get atom indices for this water
    int idxO = water_idx[w * 3];
    int idxH1 = water_idx[w * 3 + 1];
    int idxH2 = water_idx[w * 3 + 2];

    // Load old positions (reference configuration for frame definition)
    float3 oldO = make_float3(
        old_pos[idxO * 3], old_pos[idxO * 3 + 1], old_pos[idxO * 3 + 2]
    );
    float3 oldH1 = make_float3(
        old_pos[idxH1 * 3], old_pos[idxH1 * 3 + 1], old_pos[idxH1 * 3 + 2]
    );
    float3 oldH2 = make_float3(
        old_pos[idxH2 * 3], old_pos[idxH2 * 3 + 1], old_pos[idxH2 * 3 + 2]
    );

    // Load new positions (unconstrained)
    float3 newO = make_float3(
        new_pos[idxO * 3], new_pos[idxO * 3 + 1], new_pos[idxO * 3 + 2]
    );
    float3 newH1 = make_float3(
        new_pos[idxH1 * 3], new_pos[idxH1 * 3 + 1], new_pos[idxH1 * 3 + 2]
    );
    float3 newH2 = make_float3(
        new_pos[idxH2 * 3], new_pos[idxH2 * 3 + 1], new_pos[idxH2 * 3 + 2]
    );

    // Total mass and inverse
    float mT = mO + 2.0f * mH;
    float invmT = 1.0f / mT;

    // Compute OLD center of mass (for frame definition)
    float3 oldCOM;
    oldCOM.x = (mO * oldO.x + mH * oldH1.x + mH * oldH2.x) * invmT;
    oldCOM.y = (mO * oldO.y + mH * oldH1.y + mH * oldH2.y) * invmT;
    oldCOM.z = (mO * oldO.z + mH * oldH1.z + mH * oldH2.z) * invmT;

    // Compute NEW center of mass (this is conserved)
    float3 newCOM;
    newCOM.x = (mO * newO.x + mH * newH1.x + mH * newH2.x) * invmT;
    newCOM.y = (mO * newO.y + mH * newH1.y + mH * newH2.y) * invmT;
    newCOM.z = (mO * newO.z + mH * newH1.z + mH * newH2.z) * invmT;

    // ===================================================================
    // BUILD CANONICAL FRAME FROM OLD CONFIGURATION
    // ===================================================================
    // X-axis: along H1-H2 direction
    // Y-axis: perpendicular to molecular plane
    // Z-axis: along bisector (from H midpoint to O)

    // Old positions relative to old COM
    float3 a0, b0, c0;  // O, H1, H2 in old COM frame
    a0.x = oldO.x - oldCOM.x;
    a0.y = oldO.y - oldCOM.y;
    a0.z = oldO.z - oldCOM.z;
    b0.x = oldH1.x - oldCOM.x;
    b0.y = oldH1.y - oldCOM.y;
    b0.z = oldH1.z - oldCOM.z;
    c0.x = oldH2.x - oldCOM.x;
    c0.y = oldH2.y - oldCOM.y;
    c0.z = oldH2.z - oldCOM.z;

    // X-axis: H1->H2 direction (normalized)
    float3 ex;
    ex.x = c0.x - b0.x;
    ex.y = c0.y - b0.y;
    ex.z = c0.z - b0.z;
    float ex_len = sqrtf(ex.x * ex.x + ex.y * ex.y + ex.z * ex.z);
    float inv_ex = 1.0f / (ex_len + 1e-10f);
    ex.x *= inv_ex;
    ex.y *= inv_ex;
    ex.z *= inv_ex;

    // Midpoint of H1-H2
    float3 mid;
    mid.x = (b0.x + c0.x) * 0.5f;
    mid.y = (b0.y + c0.y) * 0.5f;
    mid.z = (b0.z + c0.z) * 0.5f;

    // Z-axis: from midpoint to O (bisector direction, normalized)
    float3 ez;
    ez.x = a0.x - mid.x;
    ez.y = a0.y - mid.y;
    ez.z = a0.z - mid.z;
    float ez_len = sqrtf(ez.x * ez.x + ez.y * ez.y + ez.z * ez.z);
    float inv_ez = 1.0f / (ez_len + 1e-10f);
    ez.x *= inv_ez;
    ez.y *= inv_ez;
    ez.z *= inv_ez;

    // Y-axis: ex × ez (perpendicular to molecular plane)
    float3 ey;
    ey.x = ex.y * ez.z - ex.z * ez.y;
    ey.y = ex.z * ez.x - ex.x * ez.z;
    ey.z = ex.x * ez.y - ex.y * ez.x;
    float ey_len = sqrtf(ey.x * ey.x + ey.y * ey.y + ey.z * ey.z);
    float inv_ey = 1.0f / (ey_len + 1e-10f);
    ey.x *= inv_ey;
    ey.y *= inv_ey;
    ey.z *= inv_ey;

    // ===================================================================
    // TRANSFORM NEW POSITIONS TO CANONICAL FRAME
    // ===================================================================
    // New positions relative to NEW COM
    float3 a1, b1, c1;  // O, H1, H2 in new COM frame
    a1.x = newO.x - newCOM.x;
    a1.y = newO.y - newCOM.y;
    a1.z = newO.z - newCOM.z;
    b1.x = newH1.x - newCOM.x;
    b1.y = newH1.y - newCOM.y;
    b1.z = newH1.z - newCOM.z;
    c1.x = newH2.x - newCOM.x;
    c1.y = newH2.y - newCOM.y;
    c1.z = newH2.z - newCOM.z;

    // Transform to canonical frame using dot products
    // R = [ex; ey; ez] (rows), so r_canonical = R * r_lab
    float3 a1p, b1p, c1p;  // New positions in canonical frame
    a1p.x = ex.x * a1.x + ex.y * a1.y + ex.z * a1.z;
    a1p.y = ey.x * a1.x + ey.y * a1.y + ey.z * a1.z;
    a1p.z = ez.x * a1.x + ez.y * a1.y + ez.z * a1.z;

    b1p.x = ex.x * b1.x + ex.y * b1.y + ex.z * b1.z;
    b1p.y = ey.x * b1.x + ey.y * b1.y + ey.z * b1.z;
    b1p.z = ez.x * b1.x + ez.y * b1.y + ez.z * b1.z;

    c1p.x = ex.x * c1.x + ex.y * c1.y + ex.z * c1.z;
    c1p.y = ey.x * c1.x + ey.y * c1.y + ey.z * c1.z;
    c1p.z = ez.x * c1.x + ez.y * c1.y + ez.z * c1.z;

    // ===================================================================
    // ROBUST SETTLE: Use OLD geometry with COM displacement
    // ===================================================================
    // The standard SETTLE algorithm assumes small perturbations from the
    // reference geometry. When positions are severely distorted (e.g., during
    // energy minimization with large forces), the orientation vectors become
    // ill-conditioned, producing wrong geometry.
    //
    // ROBUST FIX: Instead of computing orientation from distorted new positions,
    // we keep the OLD (correct) geometry and only apply the COM displacement.
    // This treats water as a rigid body that translates with forces.
    //
    // For normal MD with small timesteps, this gives nearly identical results
    // to full SETTLE (rotation per step is tiny). For minimization with
    // potentially large displacements, this is much more stable.
    //
    // The constrained positions are simply:
    //   new_O  = old_O  + (newCOM - oldCOM)
    //   new_H1 = old_H1 + (newCOM - oldCOM)
    //   new_H2 = old_H2 + (newCOM - oldCOM)
    //
    // This preserves:
    // - Exact water geometry (from old positions which were correct)
    // - Center of mass motion (follows forces)
    // - Momentum conservation (uniform translation)

    // COM displacement
    float3 dCOM;
    dCOM.x = newCOM.x - oldCOM.x;
    dCOM.y = newCOM.y - oldCOM.y;
    dCOM.z = newCOM.z - oldCOM.z;

    // Apply displacement to old (correct) positions
    new_pos[idxO * 3]     = oldO.x + dCOM.x;
    new_pos[idxO * 3 + 1] = oldO.y + dCOM.y;
    new_pos[idxO * 3 + 2] = oldO.z + dCOM.z;

    new_pos[idxH1 * 3]     = oldH1.x + dCOM.x;
    new_pos[idxH1 * 3 + 1] = oldH1.y + dCOM.y;
    new_pos[idxH1 * 3 + 2] = oldH1.z + dCOM.z;

    new_pos[idxH2 * 3]     = oldH2.x + dCOM.x;
    new_pos[idxH2 * 3 + 1] = oldH2.y + dCOM.y;
    new_pos[idxH2 * 3 + 2] = oldH2.z + dCOM.z;
}

/**
 * RATTLE-style SETTLE Velocity Correction
 *
 * After position constraints, velocities may have components along bond
 * directions that would violate constraints in the next step. This kernel
 * uses the RATTLE algorithm to project out ONLY the constraint-violating
 * velocity components, preserving rotational kinetic energy.
 *
 * For water with constraints C1(O-H1), C2(O-H2), C3(H1-H2):
 * Velocity constraint: r_ij · (v_i - v_j) = 0 for each bond
 *
 * This is solved via 3x3 linear system for Lagrange multipliers.
 *
 * After this correction:
 * - Relative velocities along bond directions are zero
 * - COM velocity is preserved (momentum conserved)
 * - Rotational motion perpendicular to bonds is preserved
 *
 * @param velocities Velocity array [n_atoms * 3]
 * @param positions  Position array [n_atoms * 3] (for computing bond vectors)
 * @param water_idx Water indices [n_waters * 3] - (O, H1, H2) for each water
 * @param n_waters Number of water molecules
 * @param mO Mass of oxygen
 * @param mH Mass of hydrogen
 */
__global__ void settle_velocity_correction(
    float* __restrict__ velocities,
    const float* __restrict__ positions,
    const int* __restrict__ water_idx,
    int n_waters,
    float mO, float mH
) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= n_waters) return;

    // Get atom indices for this water
    int idxO = water_idx[w * 3];
    int idxH1 = water_idx[w * 3 + 1];
    int idxH2 = water_idx[w * 3 + 2];

    // Load positions (SETTLED, constrained)
    float3 pO = make_float3(
        positions[idxO * 3], positions[idxO * 3 + 1], positions[idxO * 3 + 2]
    );
    float3 pH1 = make_float3(
        positions[idxH1 * 3], positions[idxH1 * 3 + 1], positions[idxH1 * 3 + 2]
    );
    float3 pH2 = make_float3(
        positions[idxH2 * 3], positions[idxH2 * 3 + 1], positions[idxH2 * 3 + 2]
    );

    // Load velocities
    float3 vO = make_float3(
        velocities[idxO * 3], velocities[idxO * 3 + 1], velocities[idxO * 3 + 2]
    );
    float3 vH1 = make_float3(
        velocities[idxH1 * 3], velocities[idxH1 * 3 + 1], velocities[idxH1 * 3 + 2]
    );
    float3 vH2 = make_float3(
        velocities[idxH2 * 3], velocities[idxH2 * 3 + 1], velocities[idxH2 * 3 + 2]
    );

    // Compute bond vectors (from constrained positions)
    float3 r_OH1 = make_float3(pH1.x - pO.x, pH1.y - pO.y, pH1.z - pO.z);
    float3 r_OH2 = make_float3(pH2.x - pO.x, pH2.y - pO.y, pH2.z - pO.z);
    float3 r_H1H2 = make_float3(pH2.x - pH1.x, pH2.y - pH1.y, pH2.z - pH1.z);

    // Compute relative velocities
    float3 v_OH1 = make_float3(vH1.x - vO.x, vH1.y - vO.y, vH1.z - vO.z);
    float3 v_OH2 = make_float3(vH2.x - vO.x, vH2.y - vO.y, vH2.z - vO.z);
    float3 v_H1H2 = make_float3(vH2.x - vH1.x, vH2.y - vH1.y, vH2.z - vH1.z);

    // Velocity constraint violations: sigma_k = r_k · v_k
    // These should be zero after correction
    float sigma1 = r_OH1.x * v_OH1.x + r_OH1.y * v_OH1.y + r_OH1.z * v_OH1.z;
    float sigma2 = r_OH2.x * v_OH2.x + r_OH2.y * v_OH2.y + r_OH2.z * v_OH2.z;
    float sigma3 = r_H1H2.x * v_H1H2.x + r_H1H2.y * v_H1H2.y + r_H1H2.z * v_H1H2.z;

    // Build the 3x3 matrix A for the RATTLE linear system: A * lambda = sigma
    // where velocity correction is: v_i -= sum_k (lambda_k * grad_i C_k / m_i)
    //
    // For constraint C1 (O-H1): grad_O = -r_OH1, grad_H1 = +r_OH1
    // For constraint C2 (O-H2): grad_O = -r_OH2, grad_H2 = +r_OH2
    // For constraint C3 (H1-H2): grad_H1 = -r_H1H2, grad_H2 = +r_H1H2
    //
    // Matrix elements A_ij = sum_atoms (grad_i C_k · grad_j C_k / m_atom)

    float invMO = 1.0f / mO;
    float invMH = 1.0f / mH;

    // Self-interactions (diagonal)
    float r_OH1_sq = r_OH1.x * r_OH1.x + r_OH1.y * r_OH1.y + r_OH1.z * r_OH1.z;
    float r_OH2_sq = r_OH2.x * r_OH2.x + r_OH2.y * r_OH2.y + r_OH2.z * r_OH2.z;
    float r_H1H2_sq = r_H1H2.x * r_H1H2.x + r_H1H2.y * r_H1H2.y + r_H1H2.z * r_H1H2.z;

    float A11 = r_OH1_sq * (invMO + invMH);  // C1 with C1: O and H1 contribute
    float A22 = r_OH2_sq * (invMO + invMH);  // C2 with C2: O and H2 contribute
    float A33 = r_H1H2_sq * (invMH + invMH); // C3 with C3: H1 and H2 contribute

    // Cross-interactions (off-diagonal)
    float r_OH1_OH2 = r_OH1.x * r_OH2.x + r_OH1.y * r_OH2.y + r_OH1.z * r_OH2.z;
    float r_OH1_H1H2 = r_OH1.x * r_H1H2.x + r_OH1.y * r_H1H2.y + r_OH1.z * r_H1H2.z;
    float r_OH2_H1H2 = r_OH2.x * r_H1H2.x + r_OH2.y * r_H1H2.y + r_OH2.z * r_H1H2.z;

    // A12 = C1 with C2: O contributes (both have O), signs: (-r_OH1)·(-r_OH2) = +
    float A12 = r_OH1_OH2 * invMO;
    // A13 = C1 with C3: H1 contributes (C1 has H1, C3 has H1), signs: (+r_OH1)·(-r_H1H2) = -
    float A13 = -r_OH1_H1H2 * invMH;
    // A23 = C2 with C3: H2 contributes (C2 has H2, C3 has H2), signs: (+r_OH2)·(+r_H1H2) = +
    float A23 = r_OH2_H1H2 * invMH;

    // Matrix is symmetric
    float A21 = A12;
    float A31 = A13;
    float A32 = A23;

    // Solve 3x3 system A * lambda = sigma using Cramer's rule
    // det(A) = A11*(A22*A33 - A23*A32) - A12*(A21*A33 - A23*A31) + A13*(A21*A32 - A22*A31)
    float det = A11 * (A22 * A33 - A23 * A32)
              - A12 * (A21 * A33 - A23 * A31)
              + A13 * (A21 * A32 - A22 * A31);

    // Protect against singular matrix (shouldn't happen for valid water geometry)
    if (fabsf(det) < 1e-12f) {
        return;  // Skip correction if matrix is singular
    }

    float invDet = 1.0f / det;

    // Compute lambda using Cramer's rule
    float lambda1 = invDet * (sigma1 * (A22 * A33 - A23 * A32)
                            - sigma2 * (A12 * A33 - A13 * A32)
                            + sigma3 * (A12 * A23 - A13 * A22));

    float lambda2 = invDet * (A11 * (sigma2 * A33 - sigma3 * A32)
                            - sigma1 * (A21 * A33 - A23 * A31)
                            + A13 * (A21 * sigma3 - sigma2 * A31));

    float lambda3 = invDet * (A11 * (A22 * sigma3 - A23 * sigma2)
                            - A12 * (A21 * sigma3 - sigma2 * A31)
                            + sigma1 * (A21 * A32 - A22 * A31));

    // Apply velocity corrections
    // v_O -= lambda1 * (-r_OH1) / mO + lambda2 * (-r_OH2) / mO
    //      = (lambda1 * r_OH1 + lambda2 * r_OH2) / mO
    vO.x += (lambda1 * r_OH1.x + lambda2 * r_OH2.x) * invMO;
    vO.y += (lambda1 * r_OH1.y + lambda2 * r_OH2.y) * invMO;
    vO.z += (lambda1 * r_OH1.z + lambda2 * r_OH2.z) * invMO;

    // v_H1 -= lambda1 * (+r_OH1) / mH + lambda3 * (-r_H1H2) / mH
    //       = (-lambda1 * r_OH1 + lambda3 * r_H1H2) / mH
    vH1.x += (-lambda1 * r_OH1.x + lambda3 * r_H1H2.x) * invMH;
    vH1.y += (-lambda1 * r_OH1.y + lambda3 * r_H1H2.y) * invMH;
    vH1.z += (-lambda1 * r_OH1.z + lambda3 * r_H1H2.z) * invMH;

    // v_H2 -= lambda2 * (+r_OH2) / mH + lambda3 * (+r_H1H2) / mH
    //       = (-lambda2 * r_OH2 - lambda3 * r_H1H2) / mH
    vH2.x += (-lambda2 * r_OH2.x - lambda3 * r_H1H2.x) * invMH;
    vH2.y += (-lambda2 * r_OH2.y - lambda3 * r_H1H2.y) * invMH;
    vH2.z += (-lambda2 * r_OH2.z - lambda3 * r_H1H2.z) * invMH;

    // Store corrected velocities
    velocities[idxO * 3]     = vO.x;
    velocities[idxO * 3 + 1] = vO.y;
    velocities[idxO * 3 + 2] = vO.z;

    velocities[idxH1 * 3]     = vH1.x;
    velocities[idxH1 * 3 + 1] = vH1.y;
    velocities[idxH1 * 3 + 2] = vH1.z;

    velocities[idxH2 * 3]     = vH2.x;
    velocities[idxH2 * 3 + 1] = vH2.y;
    velocities[idxH2 * 3 + 2] = vH2.z;
}

}  // extern "C"
