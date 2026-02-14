/**
 * Ligand viewer — per-ligand cards showing SMILES, dG, classification.
 *
 * Each ligand in PAYLOAD.ligand_poses has:
 *   { smiles: str, mol_block: str, dg_kcal: float, dg_error: float,
 *     classification: "NOVEL_HIT"|"FAILED_QC"|...,
 *     qed: float?, sa: float?, tanimoto: float? }
 */
const LigandViewer = (() => {
  const CLASS_BADGE = {
    NOVEL_HIT: { label: "Novel Hit", cls: "badge-hit" },
    RECAPITULATED: { label: "Recapitulated", cls: "badge-recap" },
    FAILED_QC: { label: "Failed QC", cls: "badge-fail" },
  };

  function badge(classification) {
    return CLASS_BADGE[classification] || { label: classification, cls: "badge-default" };
  }

  /** Render ligand cards into container. */
  function render(container, ligands) {
    if (!ligands || ligands.length === 0) return;
    const wrapper = document.createElement("div");
    wrapper.id = "ligand-viewer-layer";
    wrapper.className = "viewer-layer";
    wrapper.dataset.layer = "ligands";

    ligands.forEach((lig, idx) => {
      const card = document.createElement("div");
      card.className = "ligand-card";
      const b = badge(lig.classification);
      const dgStr = lig.dg_kcal !== undefined
        ? `${lig.dg_kcal.toFixed(1)} \u00b1 ${(lig.dg_error || 0).toFixed(1)} kcal/mol`
        : "N/A";

      let propsHTML = "";
      if (lig.qed !== undefined) propsHTML += `<span>QED: ${lig.qed.toFixed(2)}</span>`;
      if (lig.sa !== undefined) propsHTML += `<span>SA: ${lig.sa.toFixed(1)}</span>`;
      if (lig.tanimoto !== undefined) propsHTML += `<span>Tc: ${lig.tanimoto.toFixed(2)}</span>`;

      card.innerHTML = `
        <div class="ligand-header">
          <strong>Ligand ${idx + 1}</strong>
          <span class="badge ${b.cls}">${b.label}</span>
        </div>
        <div class="ligand-smiles mono">${lig.smiles || "—"}</div>
        <div class="ligand-dg">\u0394G<sub>bind</sub> = ${dgStr}</div>
        ${propsHTML ? `<div class="ligand-props">${propsHTML}</div>` : ""}`;
      wrapper.appendChild(card);
    });
    container.appendChild(wrapper);
  }

  return { render, badge };
})();
