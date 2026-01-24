# PRISM-Zero Validation Script for ChimeraX (Final Fix)
# Generates Top 20 Displacement Vectors & Ground Truth Comparison
# Author: PRISM-Zero System

from chimerax.core.commands import run
from chimerax.atomic import AtomicStructure
import numpy as np

def validate_prism(session):
    # 1. SETUP & LOADING
    # ------------------
    print("🚀 INITIALIZING PRISM-ZERO VALIDATION PROTOCOL...")
    run(session, "close session")
    
    # Load structures
    print("📂 Loading PDB files...")
    # Using expanduser to handle '~' correctly in python paths if needed, 
    # but ChimeraX run command handles it.
    run(session, "open ~/Downloads/6VXX.pdb")           # #1 Start
    run(session, "open ~/Downloads/covid_relaxed.pdb")  # #2 Sim
    run(session, "open ~/Downloads/6W41.pdb")           # #3 Truth

    # 2. ALIGNMENT
    # -------------------------------------
    print("⚖️  Aligning stable cores...")
    run(session, "match #2/B to #1/B")
    run(session, "match #3/C to #1/B")

    # 3. VISUALIZATION SETUP
    # ----------------------
    run(session, "hide atoms")
    run(session, "show cartoons")
    run(session, "color #1 gray")        # Start = Gray
    run(session, "color #2 cyan")        # Sim = Cyan
    run(session, "color #3 gold")        # Truth = Gold
    run(session, "transparency #3 50")   # Make Truth ghostly
    run(session, "view orient")

    # 4. MATHEMATICAL ANALYSIS
    # ------------------------
    models = session.models.list(type=AtomicStructure)
    if len(models) < 3:
        print("❌ Error: Not all models loaded.")
        return

    m_start = models[0] # 6VXX
    m_sim   = models[1] # Relaxed
    m_true  = models[2] # 6W41

    displacements = []
    
    print("\n" + "="*85)
    print(f"{'Rank':<5} | {'Residue':<10} | {'Sim Moved':<12} | {'Real Gap':<12} | {'Verdict'}")
    print("-" * 85)

    # Iterate through residues in the RBD range (Chain B)
    for r_start in m_start.residues:
        if r_start.chain_id == "B" and 319 <= r_start.number <= 541:
            try:
                # Find corresponding residues
                r_sim = m_sim.residues[m_sim.residues.numbers == r_start.number]
                r_sim = r_sim[r_sim.chain_ids == "B"][0]
                
                # 6W41 uses Chain C for RBD
                r_true = m_true.residues[m_true.residues.numbers == r_start.number]
                r_true = r_true[r_true.chain_ids == "C"]
                
                if len(r_true) == 0: continue
                r_true = r_true[0]

                # Get CA coordinates
                p_start = r_start.atoms[r_start.atoms.names == "CA"][0].scene_coord
                p_sim   = r_sim.atoms[r_sim.atoms.names == "CA"][0].scene_coord
                p_true  = r_true.atoms[r_true.atoms.names == "CA"][0].scene_coord

                # Calculate Distances
                dist_sim = np.linalg.norm(p_sim - p_start)
                err_start = np.linalg.norm(p_true - p_start)
                err_sim = np.linalg.norm(p_true - p_sim)
                
                displacements.append({
                    'res': r_start.name,
                    'num': r_start.number,
                    'dist': dist_sim,
                    'err_start': err_start,
                    'err_sim': err_sim,
                    'p1': p_start,
                    'p2': p_sim
                })

            except IndexError:
                continue

    # Sort by displacement
    displacements.sort(key=lambda x: x['dist'], reverse=True)

    # 5. REPORT & DRAW VECTORS
    # ------------------------
    for i in range(min(20, len(displacements))):
        d = displacements[i]
        
        delta = d['err_start'] - d['err_sim']
        verdict = "⚪ NEUTRAL"
        if delta > 0.5: verdict = "✅ PREDICTED"
        elif delta < -0.5: verdict = "❌ OVERSHOT"
        
        marker = ""
        if 369 <= d['num'] <= 392: marker = "⭐ TARGET"
        elif 490 <= d['num'] <= 515: marker = "🟢 LID"

        print(f"{i+1:<5} | {d['res']} {d['num']:<5} | {d['dist']:.1f} A       | {d['err_start']:.1f} -> {d['err_sim']:.1f} | {verdict} ({delta:+.1f}) {marker}")

        # Draw Vector (Arrow) - FIXED SYNTAX
        p1 = d['p1']
        p2 = d['p2']
        
        # FIX: Use 'modelId 5' instead of 'id #5'
        cmd_str = f"shape cylinder start {p1[0]:.3f},{p1[1]:.3f},{p1[2]:.3f} end {p2[0]:.3f},{p2[1]:.3f},{p2[2]:.3f} radius 0.3 color yellow modelId 5"
        run(session, cmd_str)

    print("="*85)
    
    # 6. FINAL SCENE POLISH
    run(session, "lighting soft")
    run(session, "set bg white")
    run(session, "view")
    print("✅ Validation Complete.")

validate_prism(session)
