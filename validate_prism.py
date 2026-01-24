# PRISM-Zero Validation Script for ChimeraX
# Generates Top 20 Displacement Vectors & Ground Truth Comparison
# Author: PRISM-Zero DELFICTUS I/O inc. Los Angeles, CA

from chimerax.core.commands import run
from chimerax.atomic import AtomicStructure
import numpy as np

def validate_prism(session):
    # 1. SETUP & LOADING
    # ------------------
    print("🚀 INITIALIZING PRISM-ZERO VALIDATION PROTOCOL...")
    run(session, "close session")
    
    # Load structures
    # #1: Start (Closed)
    # #2: Sim (Relaxed)
    # #3: Truth (Open Crystal Structure)
    print("📂 Loading PDB files...")
    run(session, "open ~/Downloads/6VXX.pdb")
    run(session, "open ~/Downloads/covid_relaxed.pdb")
    run(session, "open ~/Downloads/6W41.pdb")

    # 2. ALIGNMENT (Crucial for valid math)
    # -------------------------------------
    print("⚖️  Aligning stable cores...")
    # Align Sim (#2) Chain B to Start (#1) Chain B
    run(session, "match #2/B to #1/B")
    # Align Truth (#3) Chain C (RBD) to Start (#1) Chain B
    # Note: In 6W41, the RBD is often Chain C. We align it to the closed RBD.
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
    m_start = models[0] # 6VXX
    m_sim   = models[1] # Relaxed
    m_true  = models[2] # 6W41

    # Extract CA atoms for Chain B (Sim) and Chain C (Truth)
    # We focus on the RBD region (approx residues 319-541)
    
    displacements = []
    
    print("\n" + "="*85)
    print(f"{'Rank':<5} | {'Residue':<10} | {'Sim Moved':<12} | {'Real Gap':<12} | {'Verdict'}")
    print("-" * 85)

    # Iterate through residues in the RBD range
    for r_start in m_start.residues:
        if r_start.chain_id == "B" and 319 <= r_start.number <= 541:
            try:
                # Find corresponding residues in Sim and Truth
                r_sim = m_sim.residues[m_sim.residues.numbers == r_start.number]
                r_sim = r_sim[r_sim.chain_ids == "B"][0]
                
                # 6W41 usually uses Chain C for RBD
                r_true = m_true.residues[m_true.residues.numbers == r_start.number]
                r_true = r_true[r_true.chain_ids == "C"]
                
                if len(r_true) == 0: continue # Skip if not in crystal structure
                r_true = r_true[0]

                # Get CA coordinates
                p_start = r_start.atoms[r_start.atoms.names == "CA"][0].scene_coord
                p_sim   = r_sim.atoms[r_sim.atoms.names == "CA"][0].scene_coord
                p_true  = r_true.atoms[r_true.atoms.names == "CA"][0].scene_coord

                # Calculate Distances
                # How far did Sim move from Start?
                dist_sim = np.linalg.norm(p_sim - p_start)
                
                # How far was Start from Truth? (The problem to solve)
                err_start = np.linalg.norm(p_true - p_start)
                
                # How far is Sim from Truth? (Did we solve it?)
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

    # Sort by displacement (Top Movers)
    displacements.sort(key=lambda x: x['dist'], reverse=True)

    # 5. REPORT & DRAW VECTORS
    # ------------------------
    for i in range(min(20, len(displacements))):
        d = displacements[i]
        
        # Verdict Logic
        delta = d['err_start'] - d['err_sim']
        verdict = "⚪ NEUTRAL"
        if delta > 0.5: verdict = "✅ PREDICTED" # We got closer to truth
        elif delta < -0.5: verdict = "❌ OVERSHOT"  # We went too far/wrong way
        
        # Highlight Targets
        marker = ""
        if 369 <= d['num'] <= 392: marker = "⭐ TARGET"
        elif 490 <= d['num'] <= 515: marker = "🟢 LID"

        print(f"{i+1:<5} | {d['res']} {d['num']:<5} | {d['dist']:.1f} A       | {d['err_start']:.1f} -> {d['err_sim']:.1f} | {verdict} ({delta:+.1f}) {marker}")

        # Draw Vector (Arrow) in ChimeraX
        # Start (Gray) -> End (Cyan)
        p1 = d['p1']
        p2 = d['p2']
        
        # Draw Cylinder
        run(session, f"shape cylinder start {p1[0]},{p1[1]},{p1[2]} end {p2[0]},{p2[1]},{p2[2]} radius 0.3 color yellow name vector_{i}")

    print("="*85)
    
    # 6. FINAL SCENE POLISH
    # ---------------------
    run(session, "lighting soft")
    run(session, "set bg white")
    run(session, "view")
    print("✅ Validation Complete. Yellow arrows indicate predicted movement vectors.")

validate_prism(session)
