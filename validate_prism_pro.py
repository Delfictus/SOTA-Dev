# PRISM-Zero Validation Suite (Pro Visuals + Report Export)
# Generates publication-quality visuals and a forensic text report.

from chimerax.core.commands import run
from chimerax.atomic import AtomicStructure
import numpy as np
import os
from datetime import datetime

def validate_prism(session):
    # 1. INITIALIZATION
    # -----------------
    print("🚀 INITIALIZING PRISM-ZERO PRO VALIDATION...")
    run(session, "close session")
    
    # Define Paths
    home = os.path.expanduser("~")
    report_path = os.path.join(home, "Desktop", "PRISM_Validation_Report.txt")
    
    # Load Structures
    print("📂 Loading PDB files...")
    run(session, f"open {home}/Downloads/6VXX.pdb")           # #1 Start
    run(session, f"open {home}/Downloads/covid_relaxed.pdb")  # #2 Sim
    run(session, f"open {home}/Downloads/6W41.pdb")           # #3 Truth

    # 2. ALIGNMENT
    # ------------
    print("⚖️  Aligning stable cores...")
    run(session, "match #2/B to #1/B")
    run(session, "match #3/C to #1/B")

    # 3. PRO VISUALS (The "Cool" Look)
    # --------------------------------
    print("🎨 Applying Cyber-Biotech Styling...")
    run(session, "hide atoms")
    run(session, "show cartoons")
    
    # Style: Smooth tubes, shiny lighting
    run(session, "style all tube")
    run(session, "lighting soft")
    run(session, "material shiny")
    run(session, "set bg #101010") # Deep Dark Gray Background
    
    # Colors:
    # Start = Deep Steel (The Scaffold)
    run(session, "color #1 #505050") 
    
    # Sim = Neon Cyan (The Active Intelligence)
    run(session, "color #2 #00F0FF") 
    
    # Truth = Amber Gold (The Ground Truth)
    run(session, "color #3 #FFB000") 
    run(session, "transparency #3 60") # Ghostly
    
    run(session, "view orient")

    # 4. MATHEMATICAL ANALYSIS & REPORTING
    # ------------------------------------
    models = session.models.list(type=AtomicStructure)
    if len(models) < 3: return

    m_start, m_sim, m_true = models[0], models[1], models[2]
    displacements = []

    # Header for Report
    report_lines = []
    report_lines.append("=====================================================================================")
    report_lines.append(f"🧪 PRISM-ZERO VALIDATION REPORT | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=====================================================================================")
    report_lines.append(f"{'Rank':<5} | {'Residue':<10} | {'Sim Moved':<12} | {'Real Gap':<12} | {'Verdict'}")
    report_lines.append("-" * 85)

    # Iterate Chain B (RBD)
    for r_start in m_start.residues:
        if r_start.chain_id == "B" and 319 <= r_start.number <= 541:
            try:
                r_sim = m_sim.residues[(m_sim.residues.numbers == r_start.number) & (m_sim.residues.chain_ids == "B")]
                r_true = m_true.residues[(m_true.residues.numbers == r_start.number) & (m_true.residues.chain_ids == "C")]
                
                if len(r_sim) == 0 or len(r_true) == 0: continue
                
                p_start = r_start.atoms[r_start.atoms.names == "CA"][0].scene_coord
                p_sim   = r_sim[0].atoms[r_sim[0].atoms.names == "CA"][0].scene_coord
                p_true  = r_true[0].atoms[r_true[0].atoms.names == "CA"][0].scene_coord

                dist_sim = np.linalg.norm(p_sim - p_start)
                err_start = np.linalg.norm(p_true - p_start)
                err_sim = np.linalg.norm(p_true - p_sim)
                
                displacements.append({
                    'res': r_start.name, 'num': r_start.number,
                    'dist': dist_sim, 'err_start': err_start, 'err_sim': err_sim,
                    'p1': p_start, 'p2': p_sim
                })
            except: continue

    displacements.sort(key=lambda x: x['dist'], reverse=True)

    # 5. DRAW VECTORS & WRITE REPORT
    # ------------------------------
    for i in range(min(20, len(displacements))):
        d = displacements[i]
        delta = d['err_start'] - d['err_sim']
        
        verdict = "⚪ NEUTRAL"
        if delta > 0.5: verdict = "✅ PREDICTED"
        elif delta < -0.5: verdict = "❌ OVERSHOT"
        
        marker = ""
        if 369 <= d['num'] <= 392: marker = "⭐ TARGET"
        elif 490 <= d['num'] <= 515: marker = "🟢 LID"

        line = f"{i+1:<5} | {d['res']} {d['num']:<5} | {d['dist']:.1f} A       | {d['err_start']:.1f} -> {d['err_sim']:.1f} | {verdict} ({delta:+.1f}) {marker}"
        print(line)
        report_lines.append(line)

        # Draw Neon Pink Vectors
        c1 = f"{d['p1'][0]:.3f},{d['p1'][1]:.3f},{d['p1'][2]:.3f}"
        c2 = f"{d['p2'][0]:.3f},{d['p2'][1]:.3f},{d['p2'][2]:.3f}"
        try:
            run(session, f"shape cylinder start {c1} end {c2} radius 0.2 color #FF0055 id 5")
        except: pass

    # Save Report
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    
    print("="*85)
    print(f"📄 REPORT SAVED TO: {report_path}")
    print("✅ VISUALS UPDATED: Cyan=Sim, Gold=Truth, Pink=Vectors")

validate_prism(session)
