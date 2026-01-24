# PRISM-Zero Validation Suite (Corrected Reference: 6VYB)
# Fixed Unicode Error for Emojis

from chimerax.core.commands import run
from chimerax.atomic import AtomicStructure
import numpy as np
import os
from datetime import datetime

def validate_prism(session):
    print("🚀 INITIALIZING CORE-ALIGNED VALIDATION...")
    run(session, "close session")
    
    home = os.path.expanduser("~")
    report_path = os.path.join(home, "Desktop", "PRISM_Core_Report.txt")
    
    # 1. LOAD CORRECT STRUCTURES
    print("📂 Loading PDB files...")
    run(session, f"open {home}/Downloads/6VXX.pdb")           # #1 Start (Closed)
    run(session, f"open {home}/Downloads/covid_relaxed.pdb")  # #2 Sim (Your Result)
    
    print("⬇️  Fetching 6VYB (Open State Reference)...")
    run(session, "open 6VYB")                                 # #3 Truth (Open)

    # 2. ALIGNMENT
    print("⚖️  Aligning STABLE S2 CORES (Residues 900-1100)...")
    run(session, "match #2/B:900-1100 to #1/B:900-1100")
    run(session, "match #3/B:900-1100 to #1/B:900-1100")
    
    run(session, "view orient")

    # 3. VISUALS
    print("🎨 Applying Visuals...")
    run(session, "hide atoms")
    run(session, "show cartoons")
    run(session, "set bg white")
    run(session, "color #1 #C0C0C0") # Silver
    run(session, "color #2 #8B0000") # Red
    run(session, "color #3 #00FF00") # Green
    run(session, "transparency #1 50")
    run(session, "transparency #3 50")

    # 4. ANALYSIS ENGINE
    models = session.models.list(type=AtomicStructure)
    m_start, m_sim, m_true = models[0], models[1], models[2]
    displacements = []

    print("🧮 Calculating vectors...")
    target_chain = "B" 

    for r_start in m_start.residues:
        if r_start.chain_id == target_chain and 470 <= r_start.number <= 500:
            try:
                r_sim = m_sim.residues[(m_sim.residues.numbers == r_start.number) & (m_sim.residues.chain_ids == target_chain)]
                r_true = m_true.residues[(m_true.residues.numbers == r_start.number) & (m_true.residues.chain_ids == target_chain)]
                
                if len(r_sim) == 0 or len(r_true) == 0: continue
                
                p_start = r_start.atoms[r_start.atoms.names == "CA"][0].scene_coord
                p_sim   = r_sim[0].atoms[r_sim[0].atoms.names == "CA"][0].scene_coord
                p_true  = r_true[0].atoms[r_true[0].atoms.names == "CA"][0].scene_coord

                vec_ideal = p_true - p_start
                vec_sim = p_sim - p_start
                
                norm_ideal = np.linalg.norm(vec_ideal)
                norm_sim = np.linalg.norm(vec_sim)
                
                cosine = np.dot(vec_ideal, vec_sim) / (norm_ideal * norm_sim)
                
                displacements.append({
                    'res': r_start.name, 'num': r_start.number,
                    'dist_sim': norm_sim, 'dist_ideal': norm_ideal,
                    'cosine': cosine,
                    'p1': p_start, 'p2': p_sim
                })
            except: continue

    displacements.sort(key=lambda x: x['dist_sim'], reverse=True)

    # 5. REPORT
    report_lines = []
    report_lines.append(f"PRISM-ZERO CORE-ALIGNED REPORT | {datetime.now()}")
    report_lines.append(f"{'Residue':<10} | {'Moved':<10} | {'Nature':<10} | {'Direction'}")
    report_lines.append("-" * 60)
    
    for i in range(min(15, len(displacements))):
        d = displacements[i]
        
        verdict = "❓"
        if d['cosine'] > 0.8: verdict = "✅ PERFECT"
        elif d['cosine'] > 0.5: verdict = "🆗 GOOD"
        elif d['cosine'] > 0.0: verdict = "⚠️ DRIFT"
        else: verdict = "❌ WRONG WAY"
        
        line = f"{d['res']} {d['num']:<5} | {d['dist_sim']:.1f} A     | {d['dist_ideal']:.1f} A     | {verdict} ({d['cosine']:.2f})"
        print(line)
        report_lines.append(line)

        c1 = f"{d['p1'][0]:.3f},{d['p1'][1]:.3f},{d['p1'][2]:.3f}"
        c2 = f"{d['p2'][0]:.3f},{d['p2'][1]:.3f},{d['p2'][2]:.3f}"
        try:
            run(session, f"shape cylinder fromPoint {c1} toPoint {c2} radius 0.3 color #00FFFF name v{i} modelId #{500+i}")
        except: pass

    # FIX IS HERE: Added encoding="utf-8"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    print(f"✅ REPORT SAVED: {report_path}")

validate_prism(session)
