"""WT-6: Explicit Solvent Refinement + Water Maps.

Modules
-------
solvent_setup
    TIP3P/OPC solvation, ion placement, OpenMM System construction.
pocket_refinement
    Explicit-solvent MD pocket stability QC gate (STABLE/METASTABLE/COLLAPSED).
water_map_analysis
    Grid-based hydration-site thermodynamics (IST).
structural_water_finder
    Conserved water identification for downstream docking.
"""
