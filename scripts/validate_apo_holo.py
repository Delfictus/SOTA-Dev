import os
import sys

try:
    import pymol
    from pymol import cmd
except ImportError:
    print("Error: PyMOL python module not found.")
    print("Run: conda install -c schrodinger pymol")
    sys.exit(1)

def build_apo_holo_overlay():
    # Launch PyMOL in headless mode
    pymol.finish_launching(['pymol', '-cq'])

    # The static apo structure you currently have
    apo_structure = "data/targets/4lpk.pdb"
    holo_ref_id = "1pzo" 
    
    if not os.path.exists(apo_structure):
        print(f"CRITICAL FAULT: Original apo structure {apo_structure} not found.")
        sys.exit(1)

    print(f"[*] Loading Original Apo Structure: {apo_structure}")
    cmd.load(apo_structure, "apo_4lpk")

    print(f"[*] Fetching Holo-Structure ({holo_ref_id}) from RCSB PDB...")
    cmd.fetch(holo_ref_id, "holo_ref")

    # Strip water, solvent, and crystallization artifacts
    cmd.remove("solvent")
    cmd.remove("resn EDO") # Remove ethylene glycol if present
    cmd.remove("resn SO4") # Remove sulfates if present

    print("[*] Executing High-Precision RMSD Alignment...")
    # Align the apo structure to the holo reference
    cmd.align("apo_4lpk", "holo_ref")

    # --- Visually Map the Battlefield ---
    cmd.bg_color("white")

    # Format the Holo Reference (Grey protein, Magenta drug)
    cmd.color("gray70", "holo_ref")
    cmd.show("cartoon", "holo_ref")
    
    # Isolate the known cryptic pocket inhibitor (organic ligand)
    cmd.select("known_drug", "holo_ref and organic")
    cmd.show("sticks", "known_drug")
    cmd.color("magenta", "known_drug")

    # Format the Apo Prediction (Cyan protein)
    cmd.color("cyan", "apo_4lpk")
    cmd.show("cartoon", "apo_4lpk")

    # Highlight the exact Adjudicator Hotspots from the JSON Telemetry
    # HID-93, ARG-96, ASN-85, ILE-99, VAL-102, VAL-108, LEU-158
    hotspots = "85+93+96+99+102+108+158"
    cmd.select("engine_hotspots", f"apo_4lpk and resi {hotspots}")
    
    # Show the sidechains of the hotspots to see the physical blockade
    cmd.show("sticks", "engine_hotspots")
    cmd.color("yellow", "engine_hotspots")

    # Highlight ARG-96 (The massive KL=2190 trigger) specifically in Orange
    cmd.select("arg_96_trigger", "apo_4lpk and resi 96")
    cmd.color("orange", "arg_96_trigger")

    # Center the camera directly on the collision zone
    cmd.zoom("known_drug", buffer=4.0)

    # Save the polished session
    session_out = "output/4lpk_v2/apo_holo_validation.pse"
    
    # Ensure output directory exists
    os.makedirs("output/4lpk_v2", exist_ok=True)
    
    cmd.save(session_out)
    
    print(f"\n[SUCCESS] Validation session built: {session_out}")
    print("--> Open this file in your PyMOL Desktop GUI to view the overlay.")
    
    cmd.quit()

if __name__ == "__main__":
    build_apo_holo_overlay()
