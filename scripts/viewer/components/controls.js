/**
 * Layer toggle controls and export buttons.
 *
 * Toggles: surface, pocket, spikes, water map, ligands, residues, H-bonds
 * Export: PNG, PDB, SDF
 */
const Controls = (() => {
  const LAYERS = [
    { id: "pocket",   label: "Pocket Surface", icon: "\u25A0", default: true },
    { id: "spikes",   label: "Spikes",         icon: "\u25CF", default: true },
    { id: "water",    label: "Water Map",       icon: "\u25C9", default: true },
    { id: "ligands",  label: "Ligands",         icon: "\u2B23", default: true },
    { id: "residues", label: "Lining Residues", icon: "\u229A", default: true },
    { id: "hbonds",   label: "H-Bonds",         icon: "\u2500", default: false },
  ];

  const layerState = {};

  /** Build controls toolbar. */
  function render(container) {
    const toolbar = document.createElement("div");
    toolbar.id = "controls-toolbar";
    toolbar.className = "controls-toolbar";

    // Layer toggles
    const toggleGroup = document.createElement("div");
    toggleGroup.className = "toggle-group";
    LAYERS.forEach((layer) => {
      layerState[layer.id] = layer.default;
      const btn = document.createElement("button");
      btn.className = `toggle-btn ${layer.default ? "active" : ""}`;
      btn.dataset.layer = layer.id;
      btn.title = layer.label;
      btn.innerHTML = `<span class="layer-icon">${layer.icon}</span> ${layer.label}`;
      btn.addEventListener("click", () => toggleLayer(layer.id, btn));
      toggleGroup.appendChild(btn);
    });
    toolbar.appendChild(toggleGroup);

    // Export buttons
    const exportGroup = document.createElement("div");
    exportGroup.className = "export-group";
    [
      { id: "export-png", label: "PNG", fn: exportPNG },
      { id: "export-pdb", label: "PDB", fn: exportPDB },
      { id: "export-sdf", label: "SDF", fn: exportSDF },
    ].forEach(({ id, label, fn }) => {
      const btn = document.createElement("button");
      btn.id = id;
      btn.className = "export-btn";
      btn.textContent = `\u2913 ${label}`;
      btn.addEventListener("click", fn);
      exportGroup.appendChild(btn);
    });
    toolbar.appendChild(exportGroup);

    container.appendChild(toolbar);
  }

  /** Toggle layer visibility. */
  function toggleLayer(layerId, btn) {
    layerState[layerId] = !layerState[layerId];
    btn.classList.toggle("active", layerState[layerId]);

    const els = document.querySelectorAll(`[data-layer="${layerId}"]`);
    els.forEach((el) => {
      el.style.display = layerState[layerId] ? "" : "none";
    });
  }

  /** Export the viewer canvas as PNG. */
  function exportPNG() {
    const canvas = document.querySelector("#viewer-container canvas");
    if (!canvas) {
      alert("No 3D canvas found for export.");
      return;
    }
    const link = document.createElement("a");
    link.download = `${window.PAYLOAD?.target_name || "viewer"}.png`;
    link.href = canvas.toDataURL("image/png");
    link.click();
  }

  /** Export PDB structure as downloadable file. */
  function exportPDB() {
    if (!window.PAYLOAD?.pdb_structure) {
      alert("No PDB structure available.");
      return;
    }
    downloadText(
      window.PAYLOAD.pdb_structure,
      `${window.PAYLOAD.target_name || "structure"}.pdb`,
      "chemical/x-pdb"
    );
  }

  /** Export ligand SDF blocks as a combined file. */
  function exportSDF() {
    const ligands = window.PAYLOAD?.ligand_poses || [];
    if (ligands.length === 0) {
      alert("No ligand poses available.");
      return;
    }
    const sdfText = ligands
      .filter((l) => l.mol_block)
      .map((l) => l.mol_block + "\n$$$$")
      .join("\n");
    downloadText(
      sdfText,
      `${window.PAYLOAD?.target_name || "ligands"}.sdf`,
      "chemical/x-mdl-sdfile"
    );
  }

  /** Helper to download text as a file. */
  function downloadText(text, filename, mimeType) {
    const blob = new Blob([text], { type: mimeType });
    const link = document.createElement("a");
    link.download = filename;
    link.href = URL.createObjectURL(blob);
    link.click();
    URL.revokeObjectURL(link.href);
  }

  return { render, toggleLayer, exportPNG, exportPDB, exportSDF, LAYERS, layerState };
})();
