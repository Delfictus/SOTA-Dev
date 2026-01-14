/**
 * SETTLE - Analytical Constraint Solver for Rigid Water
 *
 * Implements the SETTLE algorithm for maintaining rigid TIP3P water geometry
 * during molecular dynamics. This is an analytical solution that satisfies
 * all three distance constraints (OH1, OH2, HH) exactly in a single pass.
 *
 * Reference: Miyamoto & Kollman (1992) J. Comput. Chem. 13:952-962
 *
 * Algorithm:
 * 1. Compute old and new center of mass
 * 2. Translate to COM frame
 * 3. Rotate to align with principal axes
 * 4. Solve analytical constraint equations
 * 5. Rotate back and translate to lab frame
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
 * @param ra Distance from O to COM along bisector
 * @param rb Distance from H to COM along OH (rb = rOH - ra)
 * @param rc Half of HH distance
 * @param rOH2 Target OH distance squared
 * @param rHH2 Target HH distance squared
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

    // Load old positions (reference configuration)
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

    // Compute old center of mass
    float3 oldCOM;
    oldCOM.x = (mO * oldO.x + mH * oldH1.x + mH * oldH2.x) * invmT;
    oldCOM.y = (mO * oldO.y + mH * oldH1.y + mH * oldH2.y) * invmT;
    oldCOM.z = (mO * oldO.z + mH * oldH1.z + mH * oldH2.z) * invmT;

    // Compute new center of mass
    float3 newCOM;
    newCOM.x = (mO * newO.x + mH * newH1.x + mH * newH2.x) * invmT;
    newCOM.y = (mO * newO.y + mH * newH1.y + mH * newH2.y) * invmT;
    newCOM.z = (mO * newO.z + mH * newH1.z + mH * newH2.z) * invmT;

    // Translate old positions to old COM frame
    float3 a0, b0, c0;  // Old: O, H1, H2 in COM frame
    a0.x = oldO.x - oldCOM.x;
    a0.y = oldO.y - oldCOM.y;
    a0.z = oldO.z - oldCOM.z;
    b0.x = oldH1.x - oldCOM.x;
    b0.y = oldH1.y - oldCOM.y;
    b0.z = oldH1.z - oldCOM.z;
    c0.x = oldH2.x - oldCOM.x;
    c0.y = oldH2.y - oldCOM.y;
    c0.z = oldH2.z - oldCOM.z;

    // Translate new positions to new COM frame
    float3 a1, b1, c1;  // New: O, H1, H2 in COM frame
    a1.x = newO.x - newCOM.x;
    a1.y = newO.y - newCOM.y;
    a1.z = newO.z - newCOM.z;
    b1.x = newH1.x - newCOM.x;
    b1.y = newH1.y - newCOM.y;
    b1.z = newH1.z - newCOM.z;
    c1.x = newH2.x - newCOM.x;
    c1.y = newH2.y - newCOM.y;
    c1.z = newH2.z - newCOM.z;

    // Build rotation matrix from old frame to align with Z-axis along symmetry axis
    // X-axis along H1-H2 direction, Y-axis perpendicular, Z-axis along bisector

    // Get the HH vector in old frame
    float3 hh;
    hh.x = c0.x - b0.x;
    hh.y = c0.y - b0.y;
    hh.z = c0.z - b0.z;
    float hhLen = sqrtf(hh.x * hh.x + hh.y * hh.y + hh.z * hh.z);
    float invHH = 1.0f / (hhLen + 1e-10f);
    hh.x *= invHH;
    hh.y *= invHH;
    hh.z *= invHH;

    // Get the bisector (O to midpoint of H1-H2)
    float3 mid;
    mid.x = (b0.x + c0.x) * 0.5f;
    mid.y = (b0.y + c0.y) * 0.5f;
    mid.z = (b0.z + c0.z) * 0.5f;

    float3 bisect;
    bisect.x = a0.x - mid.x;
    bisect.y = a0.y - mid.y;
    bisect.z = a0.z - mid.z;
    float bisectLen = sqrtf(bisect.x * bisect.x + bisect.y * bisect.y + bisect.z * bisect.z);
    float invBisect = 1.0f / (bisectLen + 1e-10f);
    bisect.x *= invBisect;
    bisect.y *= invBisect;
    bisect.z *= invBisect;

    // Y = bisect × hh (perpendicular)
    float3 perp;
    perp.x = bisect.y * hh.z - bisect.z * hh.y;
    perp.y = bisect.z * hh.x - bisect.x * hh.z;
    perp.z = bisect.x * hh.y - bisect.y * hh.x;
    float perpLen = sqrtf(perp.x * perp.x + perp.y * perp.y + perp.z * perp.z);
    float invPerp = 1.0f / (perpLen + 1e-10f);
    perp.x *= invPerp;
    perp.y *= invPerp;
    perp.z *= invPerp;

    // Rotation matrix: rows are hh (X), perp (Y), bisect (Z)
    // Transform new positions to canonical frame
    float3 a1p, b1p, c1p;
    a1p.x = hh.x * a1.x + hh.y * a1.y + hh.z * a1.z;
    a1p.y = perp.x * a1.x + perp.y * a1.y + perp.z * a1.z;
    a1p.z = bisect.x * a1.x + bisect.y * a1.y + bisect.z * a1.z;

    b1p.x = hh.x * b1.x + hh.y * b1.y + hh.z * b1.z;
    b1p.y = perp.x * b1.x + perp.y * b1.y + perp.z * b1.z;
    b1p.z = bisect.x * b1.x + bisect.y * b1.y + bisect.z * b1.z;

    c1p.x = hh.x * c1.x + hh.y * c1.y + hh.z * c1.z;
    c1p.y = perp.x * c1.x + perp.y * c1.y + perp.z * c1.z;
    c1p.z = bisect.x * c1.x + bisect.y * c1.y + bisect.z * c1.z;

    // In canonical frame, the constrained positions are:
    // O: (0, 0, -ra)
    // H1: (-rc, 0, rb)
    // H2: (+rc, 0, rb)
    // where ra, rb, rc are the TIP3P geometry parameters

    // SETTLE analytical solution:
    // We need to find a rotation that maps the unconstrained positions
    // to positions satisfying the constraints.

    // Compute the rotation angle and axis needed
    // Using the SETTLE formulation from Miyamoto & Kollman

    // Intermediate values
    float sinphi = a1p.z / ra;  // sin of tilt angle
    if (sinphi > 1.0f) sinphi = 1.0f;
    if (sinphi < -1.0f) sinphi = -1.0f;
    float cosphi = sqrtf(1.0f - sinphi * sinphi);

    float sinpsi = (b1p.z - c1p.z) / (2.0f * rc * cosphi + 1e-10f);
    if (sinpsi > 1.0f) sinpsi = 1.0f;
    if (sinpsi < -1.0f) sinpsi = -1.0f;
    float cospsi = sqrtf(1.0f - sinpsi * sinpsi);

    // Constrained positions in canonical frame
    float3 a2p, b2p, c2p;

    // O at (0, ra*sin(theta), -ra*cos(theta)) where theta=0 for canonical
    a2p.x = 0.0f;
    a2p.y = ra * sinphi;
    a2p.z = -ra * cosphi;

    // H1 at (-rc*cos(psi), rb*sin(phi)-rc*sin(psi)*cos(phi), rb*cos(phi)+rc*sin(psi)*sin(phi))
    b2p.x = -rc * cospsi;
    b2p.y = rb * sinphi - rc * sinpsi * cosphi;
    b2p.z = rb * cosphi + rc * sinpsi * sinphi;

    // H2 at (+rc*cos(psi), rb*sin(phi)+rc*sin(psi)*cos(phi), rb*cos(phi)-rc*sin(psi)*sin(phi))
    c2p.x = rc * cospsi;
    c2p.y = rb * sinphi + rc * sinpsi * cosphi;
    c2p.z = rb * cosphi - rc * sinpsi * sinphi;

    // Rotate back from canonical frame to lab frame
    // Inverse rotation: columns are hh (X), perp (Y), bisect (Z)
    float3 a2, b2, c2;
    a2.x = hh.x * a2p.x + perp.x * a2p.y + bisect.x * a2p.z;
    a2.y = hh.y * a2p.x + perp.y * a2p.y + bisect.y * a2p.z;
    a2.z = hh.z * a2p.x + perp.z * a2p.y + bisect.z * a2p.z;

    b2.x = hh.x * b2p.x + perp.x * b2p.y + bisect.x * b2p.z;
    b2.y = hh.y * b2p.x + perp.y * b2p.y + bisect.y * b2p.z;
    b2.z = hh.z * b2p.x + perp.z * b2p.y + bisect.z * b2p.z;

    c2.x = hh.x * c2p.x + perp.x * c2p.y + bisect.x * c2p.z;
    c2.y = hh.y * c2p.x + perp.y * c2p.y + bisect.y * c2p.z;
    c2.z = hh.z * c2p.x + perp.z * c2p.y + bisect.z * c2p.z;

    // Translate back to lab frame (using new COM)
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
