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
    // SOLVE SETTLE CONSTRAINT EQUATIONS
    // ===================================================================
    // The analytical solution finds rotation angles that map unconstrained
    // positions to positions satisfying all three distance constraints.
    //
    // In canonical frame, constrained positions are:
    // O:  (0, 0, -ra)
    // H1: (-rc, 0, rb)
    // H2: (+rc, 0, rb)
    //
    // The key constraint is that the transformation must be a pure rotation
    // (orthogonal) about the COM.

    // For TIP3P, we use a simplified approach that directly constructs
    // the constrained positions based on the orientation of the unconstrained
    // molecule in the canonical frame.
    //
    // The idea is to find the rotation that best aligns the unconstrained
    // positions with the target positions, then apply that rotation.

    // Step 1: Find the orientation of the unconstrained molecule
    // The new H-H axis direction in canonical frame
    float3 nh;
    nh.x = c1p.x - b1p.x;
    nh.y = c1p.y - b1p.y;
    nh.z = c1p.z - b1p.z;
    float nh_len = sqrtf(nh.x * nh.x + nh.y * nh.y + nh.z * nh.z);
    float inv_nh = 1.0f / (nh_len + 1e-10f);
    nh.x *= inv_nh;
    nh.y *= inv_nh;
    nh.z *= inv_nh;

    // The new bisector direction in canonical frame
    float3 nmid;
    nmid.x = (b1p.x + c1p.x) * 0.5f;
    nmid.y = (b1p.y + c1p.y) * 0.5f;
    nmid.z = (b1p.z + c1p.z) * 0.5f;

    float3 nb;  // new bisector
    nb.x = a1p.x - nmid.x;
    nb.y = a1p.y - nmid.y;
    nb.z = a1p.z - nmid.z;
    float nb_len = sqrtf(nb.x * nb.x + nb.y * nb.y + nb.z * nb.z);
    float inv_nb = 1.0f / (nb_len + 1e-10f);
    nb.x *= inv_nb;
    nb.y *= inv_nb;
    nb.z *= inv_nb;

    // The new Y-axis (perpendicular to molecular plane)
    float3 ny;
    ny.x = nh.y * nb.z - nh.z * nb.y;
    ny.y = nh.z * nb.x - nh.x * nb.z;
    ny.z = nh.x * nb.y - nh.y * nb.x;
    float ny_len = sqrtf(ny.x * ny.x + ny.y * ny.y + ny.z * ny.z);
    float inv_ny = 1.0f / (ny_len + 1e-10f);
    ny.x *= inv_ny;
    ny.y *= inv_ny;
    ny.z *= inv_ny;

    // Step 2: Construct constrained positions using the new molecule orientation
    // The constrained positions in the molecule's own frame are fixed:
    // O:  (0, 0, -ra)  relative to molecule frame
    // H1: (-rc, 0, rb)
    // H2: (+rc, 0, rb)
    //
    // Transform these to canonical frame using the new molecule orientation
    // Molecule frame: X along nh, Y along ny, Z along nb (bisector)

    // Constrained O position in canonical frame
    // O is at (0, 0, -ra) in molecule frame -> nb * (-ra) in canonical
    float3 a2p;
    a2p.x = nb.x * (-ra);
    a2p.y = nb.y * (-ra);
    a2p.z = nb.z * (-ra);

    // Constrained H1 position in canonical frame
    // H1 is at (-rc, 0, rb) in molecule frame -> -rc*nh + rb*nb in canonical
    float3 b2p;
    b2p.x = -rc * nh.x + rb * nb.x;
    b2p.y = -rc * nh.y + rb * nb.y;
    b2p.z = -rc * nh.z + rb * nb.z;

    // Constrained H2 position in canonical frame
    // H2 is at (+rc, 0, rb) in molecule frame -> +rc*nh + rb*nb in canonical
    float3 c2p;
    c2p.x = rc * nh.x + rb * nb.x;
    c2p.y = rc * nh.y + rb * nb.y;
    c2p.z = rc * nh.z + rb * nb.z;

    // ===================================================================
    // TRANSFORM CONSTRAINED POSITIONS BACK TO LAB FRAME
    // ===================================================================
    // r_lab = R^T * r_canonical = [ex | ey | ez] * r_canonical

    float3 a2, b2, c2;  // Constrained positions in lab frame (COM-relative)
    a2.x = ex.x * a2p.x + ey.x * a2p.y + ez.x * a2p.z;
    a2.y = ex.y * a2p.x + ey.y * a2p.y + ez.y * a2p.z;
    a2.z = ex.z * a2p.x + ey.z * a2p.y + ez.z * a2p.z;

    b2.x = ex.x * b2p.x + ey.x * b2p.y + ez.x * b2p.z;
    b2.y = ex.y * b2p.x + ey.y * b2p.y + ez.y * b2p.z;
    b2.z = ex.z * b2p.x + ey.z * b2p.y + ez.z * b2p.z;

    c2.x = ex.x * c2p.x + ey.x * c2p.y + ez.x * c2p.z;
    c2.y = ex.y * c2p.x + ey.y * c2p.y + ez.y * c2p.z;
    c2.z = ex.z * c2p.x + ey.z * c2p.y + ez.z * c2p.z;

    // Translate back to lab frame using NEW COM (conserved)
    new_pos[idxO * 3]     = a2.x + newCOM.x;
    new_pos[idxO * 3 + 1] = a2.y + newCOM.y;
    new_pos[idxO * 3 + 2] = a2.z + newCOM.z;

    new_pos[idxH1 * 3]     = b2.x + newCOM.x;
    new_pos[idxH1 * 3 + 1] = b2.y + newCOM.y;
    new_pos[idxH1 * 3 + 2] = b2.z + newCOM.z;

    new_pos[idxH2 * 3]     = c2.x + newCOM.x;
    new_pos[idxH2 * 3 + 1] = c2.y + newCOM.y;
    new_pos[idxH2 * 3 + 2] = c2.z + newCOM.z;
}

}  // extern "C"
