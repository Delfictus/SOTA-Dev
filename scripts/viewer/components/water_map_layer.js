/**
 * Water map layer — hydration site spheres colored by classification.
 *
 * Each site in PAYLOAD.water_map_sites has:
 *   { position: [x,y,z], occupancy: 0-1, delta_g: float,
 *     classification: "CONSERVED_HAPPY"|"CONSERVED_UNHAPPY"|"BULK",
 *     color: "red"|"blue"|... }
 *
 * Red = displaceable (unhappy, dG > +1.0)
 * Blue = conserved (happy, dG < -1.0)
 */
const WaterMapLayer = (() => {
  const CLASS_COLORS = {
    CONSERVED_HAPPY: "#2196f3",     // blue
    CONSERVED_UNHAPPY: "#f44336",   // red
    BULK: "#9e9e9e",                // grey
  };

  function getColor(classification) {
    return CLASS_COLORS[classification] || "#9e9e9e";
  }

  /** Render water map sites as a list/table in the info panel. */
  function render(container, sites) {
    if (!sites || sites.length === 0) return;
    const wrapper = document.createElement("div");
    wrapper.id = "water-map-layer";
    wrapper.className = "viewer-layer";
    wrapper.dataset.layer = "water";

    const nHappy = sites.filter(s => s.classification === "CONSERVED_HAPPY").length;
    const nUnhappy = sites.filter(s => s.classification === "CONSERVED_UNHAPPY").length;
    const nBulk = sites.filter(s => s.classification === "BULK").length;

    const summary = document.createElement("div");
    summary.className = "water-summary";
    summary.innerHTML = `
      <span class="water-badge happy">${nHappy} conserved</span>
      <span class="water-badge unhappy">${nUnhappy} displaceable</span>
      ${nBulk > 0 ? `<span class="water-badge bulk">${nBulk} bulk</span>` : ""}`;
    wrapper.appendChild(summary);

    const table = document.createElement("table");
    table.className = "water-table";
    table.innerHTML = `
      <thead><tr>
        <th>Class</th><th>\u0394G (kcal/mol)</th><th>Occupancy</th><th>Position</th>
      </tr></thead>`;
    const tbody = document.createElement("tbody");

    sites.forEach((s) => {
      const tr = document.createElement("tr");
      const color = getColor(s.classification);
      const pos = s.position || [0,0,0];
      tr.innerHTML = `
        <td><span class="water-dot" style="background:${color}"></span>${s.classification}</td>
        <td class="mono">${(s.delta_g >= 0 ? "+" : "") + s.delta_g.toFixed(1)}</td>
        <td>${(s.occupancy * 100).toFixed(0)}%</td>
        <td class="mono">${pos.map(v => v.toFixed(1)).join(", ")}</td>`;
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    wrapper.appendChild(table);
    container.appendChild(wrapper);
  }

  return { render, getColor, CLASS_COLORS };
})();
