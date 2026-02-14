/**
 * Report panel — collapsible side panel with pocket-level summary.
 *
 * Shows: target name, P_open, druggability class, displaceable waters,
 * quality score, and metadata.
 */
const ReportPanel = (() => {
  /** Format P_open as percentage with classification. */
  function classifyPopen(pOpen) {
    if (pOpen === null || pOpen === undefined) return { text: "N/A", cls: "popen-na" };
    const pct = (pOpen * 100).toFixed(0);
    if (pOpen >= 0.5) return { text: `${pct}% (Stable Open)`, cls: "popen-stable" };
    if (pOpen >= 0.1) return { text: `${pct}% (Transient)`, cls: "popen-transient" };
    return { text: `${pct}% (Rare Event)`, cls: "popen-rare" };
  }

  /** Build the report panel DOM. */
  function render(container, payload) {
    const panel = document.createElement("div");
    panel.id = "report-panel";
    panel.className = "report-panel";

    const pOpen = classifyPopen(payload.p_open);
    const meta = payload.metadata || {};

    panel.innerHTML = `
      <div class="report-header" onclick="ReportPanel.toggle()">
        <h2>${payload.target_name || "Unknown Target"}</h2>
        <span class="collapse-icon" id="collapse-icon">\u25C0</span>
      </div>
      <div class="report-body" id="report-body">
        <section class="report-section">
          <h3>Pocket Summary</h3>
          <dl>
            <dt>P<sub>open</sub></dt>
            <dd class="${pOpen.cls}">${pOpen.text}</dd>
            <dt>Druggability</dt>
            <dd>${meta.druggability_class || "—"}</dd>
            <dt>Displaceable Waters</dt>
            <dd>${meta.n_displaceable_waters !== undefined ? meta.n_displaceable_waters : "—"}</dd>
            <dt>Lining Residues</dt>
            <dd class="mono">${(payload.lining_residues || []).join(", ") || "—"}</dd>
            <dt>QA Score</dt>
            <dd>${meta.qa_score !== undefined ? meta.qa_score.toFixed(2) : "—"}</dd>
          </dl>
        </section>
        <section class="report-section">
          <h3>Ligands (${(payload.ligand_poses || []).length})</h3>
          <div id="ligand-detail-slot"></div>
        </section>
        <section class="report-section">
          <h3>Spikes (${(payload.spike_positions || []).length})</h3>
          <div id="spike-detail-slot"></div>
        </section>
        <section class="report-section">
          <h3>Water Map (${(payload.water_map_sites || []).length} sites)</h3>
          <div id="water-detail-slot"></div>
        </section>
      </div>`;
    container.appendChild(panel);
  }

  /** Toggle panel collapsed/expanded. */
  function toggle() {
    const body = document.getElementById("report-body");
    const icon = document.getElementById("collapse-icon");
    if (!body) return;
    const collapsed = body.style.display === "none";
    body.style.display = collapsed ? "block" : "none";
    if (icon) icon.textContent = collapsed ? "\u25C0" : "\u25B6";
  }

  return { render, toggle, classifyPopen };
})();
