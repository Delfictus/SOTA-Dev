/**
 * Spike position overlay — renders pharmacophore spike markers.
 *
 * Each spike in PAYLOAD.spike_positions has:
 *   { position: [x,y,z], type: "BNZ"|"TYR"|..., intensity: 0-1, residue: "TYR142" }
 *
 * Spikes are rendered as colored markers in the 3D scene.
 */
const SpikeOverlay = (() => {
  const SPIKE_COLORS = {
    BNZ: "#ff6b35",   // aromatic — orange
    TYR: "#e63946",   // tyrosine — red
    CATION: "#457b9d", // cation — steel blue
    ANION: "#e9c46a",  // anion — yellow
    HBD: "#2a9d8f",    // H-bond donor — teal
    HBA: "#264653",    // H-bond acceptor — dark teal
    HY: "#a8dadc",     // hydrophobic — light blue
  };

  function getColor(spikeType) {
    return SPIKE_COLORS[spikeType] || "#888888";
  }

  /** Build spike position table in the info panel. */
  function render(container, spikes) {
    if (!spikes || spikes.length === 0) return;
    const wrapper = document.createElement("div");
    wrapper.id = "spike-overlay-layer";
    wrapper.className = "viewer-layer";
    wrapper.dataset.layer = "spikes";

    const table = document.createElement("table");
    table.className = "spike-table";
    table.innerHTML = `
      <thead><tr>
        <th>Type</th><th>Residue</th><th>Intensity</th><th>Position</th>
      </tr></thead>`;
    const tbody = document.createElement("tbody");

    spikes.forEach((s) => {
      const tr = document.createElement("tr");
      const color = getColor(s.type);
      const pos = s.position || [0,0,0];
      tr.innerHTML = `
        <td><span class="spike-dot" style="background:${color}"></span>${s.type}</td>
        <td>${s.residue || "—"}</td>
        <td>${(s.intensity * 100).toFixed(0)}%</td>
        <td class="mono">${pos.map(v => v.toFixed(1)).join(", ")}</td>`;
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    wrapper.appendChild(table);
    container.appendChild(wrapper);
  }

  return { render, getColor, SPIKE_COLORS };
})();
