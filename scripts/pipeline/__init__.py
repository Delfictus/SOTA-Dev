"""WT-4: End-to-end PRISM-4D pipeline orchestrator + reporting.

Wires all worktree deliverables into a single-command pipeline:
PDB -> novel validated hits with publication-quality reports.

Modules:
    prism_fep_pipeline  Master orchestrator (16-stage pipeline).
    audit_trail         Anti-leakage audit logging.
    report_generator    Publication-quality markdown reports.
"""
