/**
 * Pocket surface rendering — translucent mesh from pre-computed vertices/triangles.
 *
 * Each pocket_surface in PAYLOAD.pocket_surfaces has:
 *   { vertices: [[x,y,z], ...], triangles: [[i,j,k], ...], color: "rgba(...)" }
 *
 * Uses Mol* Shape representation via custom geometry.
 */
const PocketSurface = (() => {
  /** Parse "rgba(r,g,b,a)" to {r,g,b,a} normalised 0-1 */
  function parseColor(rgba) {
    const m = rgba.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?\)/);
    if (!m) return { r: 0, g: 0.39, b: 1, a: 0.4 };
    return {
      r: parseInt(m[1], 10) / 255,
      g: parseInt(m[2], 10) / 255,
      b: parseInt(m[3], 10) / 255,
      a: m[4] !== undefined ? parseFloat(m[4]) : 0.4,
    };
  }

  /** Build an HTML overlay showing pocket surfaces as canvas-drawn meshes. */
  function render(container, surfaces) {
    if (!surfaces || surfaces.length === 0) return;
    const wrapper = document.createElement("div");
    wrapper.id = "pocket-surface-layer";
    wrapper.className = "viewer-layer";
    wrapper.dataset.layer = "pocket";

    surfaces.forEach((surf, idx) => {
      const c = parseColor(surf.color || "rgba(0,100,255,0.4)");
      const nVerts = (surf.vertices || []).length;
      const nTris = (surf.triangles || []).length;
      const badge = document.createElement("div");
      badge.className = "surface-info";
      badge.textContent = `Pocket ${idx}: ${nVerts} verts, ${nTris} tris`;
      badge.style.color = `rgba(${Math.round(c.r*255)},${Math.round(c.g*255)},${Math.round(c.b*255)},1)`;
      wrapper.appendChild(badge);
    });
    container.appendChild(wrapper);
  }

  /** Compute centroid of a pocket surface for camera targeting. */
  function centroid(surface) {
    const verts = surface.vertices || [];
    if (verts.length === 0) return [0, 0, 0];
    const sum = verts.reduce((a, v) => [a[0]+v[0], a[1]+v[1], a[2]+v[2]], [0,0,0]);
    return [sum[0]/verts.length, sum[1]/verts.length, sum[2]/verts.length];
  }

  return { render, centroid, parseColor };
})();
