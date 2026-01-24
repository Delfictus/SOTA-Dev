# PRISM-Zero Validation Suite (Diagnostic "No Fail" Edition)
from chimerax.core.commands import run
from chimerax.atomic import AtomicStructure
import numpy as np
import os
from datetime import datetime

def validate_prism(session):
    print("\n" + "="*50)
    print("🚀 STARTING DIAGNOSTIC VALIDATION RUN")
    print("="*50)
    
    # 1. SETUP PATHS
    home = os.path.expanduser("~")
    report_path = os.path.join(home, "Desktop", "PRISM_Validation_Report.txt")
    
    files = [
        f"{home}/Downloads/6VXX.pdb",          # Start
        f"{home}/Downloads/covid_relaxed.pdb", # Sim
        f"{home}/Downloads/6W41.pdb"           # Truth
    ]

    # 2. CHECK FILES EXIST
    print("📂 Checking files...")
    missing = False
    for f in files:
        if os.path.exists(f):
            print(f"  ✅ Found: {os.path.basename(f)}")
        else:
            print(f"  ❌ MISSING: {f}")
            missing = True
    
    if missing:
        print("🛑 STOPPING: Cannot proceed without all files.")
        return

    # 3. RESET & LOAD
    print("🔄 Resetting session...")
    run(session, "close session")
    
    for f in files:
        run(session, f"open {f}")

    # 4. VERIFY MODELS
    models = session.models.list(type=AtomicStructure)
    print(f"📊 Models Loaded: {len(models)}")
    
    if len(models) < 3:
        print("❌ ERROR: Expected 3 models, found fewer.")
        print("   Make sure the PDB files are valid.")
        return

    m_start, m_sim, m_true = models[0], models[1], models[2]
    print(f"   Model #1 (Start): {m_start.name} ({len(m_start.residues)} residues)")
    print(f"   Model #2 (Sim)  : {m_sim.name} ({len(m_sim.residues)} residues)")
    print(f"   Model #3 (Truth): {m_true.name} ({len(m_true.residues)} residues)")

    # 5. ALIGNMENT
    print("⚖️  Aligning structures...")
    run(session, "match #2/B to #1/B")
    run(session, "match #3/C to #1/B")
    run(session, "view orient")

    # 6. CALCULATIONS
    print("🧮 Calculating vectors...")
    displacements = []
    
    for r_start in m_start.residues:
        if r_start.chain_id == "B" and 319 <= r_start.number <= 541:
            try:
                r_sim = m_sim.residues[(m_sim.residues.numbers == r_start.number) & (m_sim.residues.chain_ids == "B")]
                r_true = m_true.residues[(m_true.residues.numbers == r_start.number) & (m_true.residues.chain_ids == "C")]
                
                if len(r_sim) == 0 or len(r_true) == 0: continue
                
                p_start = r_start.atoms[r_start.atoms.names == "CA"][0].scene_coord
                p_sim   = r_sim[0].atoms[r_sim[0].atoms.names == "CA"][0].scene_coord
                p_true  = r_true[0].atoms[r_true[0].atoms.names == "CA"][0].scene_coord

                dist = np.linalg.norm(p_sim - p_start)
                err_start = np.linalg.norm(p_true - p_start)
                err_sim = np.linalg.norm(p_true - p_sim)
                
                displacements.append({
                    'res': r_start.name, 'num': r_start.number,
                    'dist': dist, 'err_start': err_start, 'err_sim': err_sim,
                    'p1': p_start, 'p2': p_sim
                })
            except: continue

    print(f"✅ Calculated {len(displacements)} vectors.")
    displacements.sort(key=lambda x: x['dist'], reverse=True)

    # 7. GENERATE REPORT (Forced Write)
    print("📝 Writing report...")
    report_lines = []
    report_lines.append(f"PRISM-ZERO DIAGNOSTIC REPORT | {datetime.now()}")
    report_lines.append("=================================================================================")
    report_lines.append(f"{'Rank':<5} | {'Residue':<10} | {'Moved':<10} | {'Goal Dist':<12} | {'Final Dist':<12} | {'Verdict'}")
    report_lines.append("-" * 90)

    count = 0
    for i in range(min(20, len(displacements))):
        d = displacements[i]
        delta = d['err_start'] - d['err_sim']
        verdict = "✅ PREDICTED" if delta > 0.5 else "❌ OVERSHOT" if delta < -0.5 else "⚪ NEUTRAL"
        
        line = f"{i+1:<5} | {d['res']} {d['num']:<5} | {d['dist']:.1f} A     | {d['err_start']:.1f} A       | {d['err_sim']:.1f} A       | {verdict} ({delta:+.1f})"
        report_lines.append(line)
        count += 1

        # Draw Vector (Safe Mode)
        c1 = f"{d['p1'][0]:.3f},{d['p1'][1]:.3f},{d['p1'][2]:.3f}"
        c2 = f"{d['p2'][0]:.3f},{d['p2'][1]:.3f},{d['p2'][2]:.3f}"
        try:
            run(session, f"shape cylinder fromPoint {c1} toPoint {c2} radius 0.3 color #00FFFF name v{i} modelId #{500+i}")
        except: pass

    if count == 0:
        report_lines.append("⚠️ NO DATA: No matching residues found between the structures.")

    try:
        with open(report_path, "w") as f:
            f.write("\n".join(report_lines))
        print(f"🎉 SUCCESS: Report written to {report_path}")
    except Exception as e:
        print(f"❌ WRITE ERROR: Could not write file. {e}")
        # Print to log as backup
        print("\n".join(report_lines))

validate_prism(session)
