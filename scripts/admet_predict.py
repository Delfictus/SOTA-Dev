#!/usr/bin/env python3
"""
PRISM4D ADMET & Drug-Likeness Prediction
==========================================
Computes Lipinski/Veber/Ghose rules, QED (Quantitative Estimate of Drug-likeness),
TPSA, LogP, and basic ADMET flags for docked ligands.

Input:  Docked SDF files (from GNINA/UniDock output)
Output: JSON + markdown table with drug-likeness profile per ligand

References:
    Lipinski et al. Adv Drug Deliv Rev 2001. doi:10.1016/S0169-409X(00)00129-0
    Veber et al. J Med Chem 2002. doi:10.1021/jm020017n
    Bickerton et al. Nat Chem 2012. doi:10.1038/nchem.1243 (QED)
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, QED, Crippen, rdMolDescriptors


def compute_admet(mol, name="Unknown"):
    """Compute ADMET properties for a single molecule."""
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hba = Descriptors.NumHAcceptors(mol)
    hbd = Descriptors.NumHDonors(mol)
    tpsa = Descriptors.TPSA(mol)
    rotbonds = Descriptors.NumRotatableBonds(mol)
    rings = Descriptors.RingCount(mol)
    aromatic_rings = Descriptors.NumAromaticRings(mol)
    heavy_atoms = mol.GetNumHeavyAtoms()
    fsp3 = rdMolDescriptors.CalcFractionCSP3(mol)
    qed_score = QED.qed(mol)

    # Lipinski Rule of 5
    lipinski_violations = 0
    if mw > 500: lipinski_violations += 1
    if logp > 5: lipinski_violations += 1
    if hba > 10: lipinski_violations += 1
    if hbd > 5: lipinski_violations += 1
    lipinski_pass = lipinski_violations <= 1

    # Veber rules (oral bioavailability)
    veber_pass = tpsa <= 140 and rotbonds <= 10

    # Ghose filter
    ghose_pass = (160 <= mw <= 480 and -0.4 <= logp <= 5.6 and
                  40 <= heavy_atoms <= 130 and 20 <= Descriptors.MolMR(mol) <= 130)

    # PAINS check (basic — count known promiscuous substructures)
    # Simplified: flag if too many Michael acceptors or quinones
    smarts_pains = [
        "[#6]=[#6]-[#6]=[O,S]",  # Michael acceptor
        "c1cc(=O)c(=O)cc1",      # quinone
    ]
    pains_flags = 0
    for smarts in smarts_pains:
        pat = Chem.MolFromSmarts(smarts)
        if pat and mol.HasSubstructMatch(pat):
            pains_flags += 1

    # Solubility estimate (ESOL: Delaney, JCICS 2004)
    # logS = 0.16 - 0.63*cLogP - 0.0062*MW + 0.066*RB - 0.74*AP
    ap = aromatic_rings / max(rings, 1) if rings > 0 else 0
    log_s = 0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * rotbonds - 0.74 * ap

    # Classification
    if log_s >= -2:
        sol_class = "Highly soluble"
    elif log_s >= -4:
        sol_class = "Soluble"
    elif log_s >= -6:
        sol_class = "Moderately soluble"
    else:
        sol_class = "Poorly soluble"

    # BBB permeability estimate (simple: MW<450, TPSA<90, logP 1-3)
    bbb_likely = mw < 450 and tpsa < 90 and 1 <= logp <= 3

    # CYP liability flags
    # High logP + aromatic rings → likely CYP3A4 substrate
    cyp_risk = "High" if logp > 3 and aromatic_rings >= 2 else "Low" if logp < 2 else "Moderate"

    return {
        "name": name,
        "molecular_weight": round(mw, 1),
        "cLogP": round(logp, 2),
        "HBA": hba,
        "HBD": hbd,
        "TPSA": round(tpsa, 1),
        "rotatable_bonds": rotbonds,
        "rings": rings,
        "aromatic_rings": aromatic_rings,
        "heavy_atoms": heavy_atoms,
        "Fsp3": round(fsp3, 2),
        "QED": round(qed_score, 3),
        "lipinski_violations": lipinski_violations,
        "lipinski_pass": lipinski_pass,
        "veber_pass": veber_pass,
        "ghose_pass": ghose_pass,
        "PAINS_flags": pains_flags,
        "ESOL_logS": round(log_s, 2),
        "solubility_class": sol_class,
        "BBB_permeable": bbb_likely,
        "CYP_risk": cyp_risk,
        "SMILES": Chem.MolToSmiles(mol),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
    }


def generate_admet_report(results, output_md, output_json):
    """Generate markdown and JSON ADMET reports."""
    # JSON
    with open(output_json, "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(), "ligands": results}, f, indent=2)

    # Markdown
    with open(output_md, "w") as f:
        f.write("# PRISM4D ADMET & Drug-Likeness Report\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

        # Summary table
        f.write("## Drug-Likeness Summary\n\n")
        f.write("| Ligand | MW | cLogP | HBA | HBD | TPSA | RotB | QED | Lipinski | Veber |\n")
        f.write("|--------|-----|-------|-----|-----|------|------|-----|----------|-------|\n")
        for r in results:
            lip = "PASS" if r["lipinski_pass"] else f"FAIL ({r['lipinski_violations']})"
            veb = "PASS" if r["veber_pass"] else "FAIL"
            f.write(f"| {r['name']} | {r['molecular_weight']} | {r['cLogP']} | "
                    f"{r['HBA']} | {r['HBD']} | {r['TPSA']} | {r['rotatable_bonds']} | "
                    f"{r['QED']:.2f} | {lip} | {veb} |\n")
        f.write("\n")

        # ADMET flags
        f.write("## ADMET Flags\n\n")
        f.write("| Ligand | Solubility | BBB | CYP Risk | PAINS | Fsp3 | Ghose |\n")
        f.write("|--------|-----------|-----|----------|-------|------|-------|\n")
        for r in results:
            bbb = "Likely" if r["BBB_permeable"] else "Unlikely"
            ghose = "PASS" if r["ghose_pass"] else "FAIL"
            pains = "Clean" if r["PAINS_flags"] == 0 else f"{r['PAINS_flags']} flag(s)"
            f.write(f"| {r['name']} | {r['solubility_class']} | {bbb} | "
                    f"{r['CYP_risk']} | {pains} | {r['Fsp3']} | {ghose} |\n")
        f.write("\n")

        # Detailed per-ligand
        f.write("## Detailed Profiles\n\n")
        for r in results:
            f.write(f"### {r['name']}\n\n")
            f.write(f"- **Formula**: {r['formula']}\n")
            f.write(f"- **SMILES**: `{r['SMILES']}`\n")
            f.write(f"- **MW**: {r['molecular_weight']} Da\n")
            f.write(f"- **cLogP**: {r['cLogP']}\n")
            f.write(f"- **TPSA**: {r['TPSA']} A^2\n")
            f.write(f"- **QED**: {r['QED']:.3f} (0=worst, 1=best)\n")
            f.write(f"- **Lipinski violations**: {r['lipinski_violations']} ")
            if r['lipinski_pass']:
                f.write("(PASS)\n")
            else:
                f.write("(FAIL — not orally bioavailable by Ro5)\n")
            f.write(f"- **Rotatable bonds**: {r['rotatable_bonds']} ")
            f.write("(OK)\n" if r['rotatable_bonds'] <= 10 else "(HIGH — poor oral absorption)\n")
            f.write(f"- **Aromatic rings**: {r['aromatic_rings']}, Fsp3: {r['Fsp3']}\n")
            f.write(f"- **Solubility**: {r['solubility_class']} (ESOL logS = {r['ESOL_logS']})\n")
            f.write(f"- **BBB permeability**: {'Likely' if r['BBB_permeable'] else 'Unlikely'}\n")
            f.write(f"- **CYP liability**: {r['CYP_risk']}\n")
            f.write(f"- **PAINS**: {r['PAINS_flags']} alerts\n\n")

        # Interpretation guide
        f.write("## Interpretation Guide\n\n")
        f.write("| Metric | Threshold | Meaning |\n")
        f.write("|--------|-----------|--------|\n")
        f.write("| QED | >0.5 good, >0.7 excellent | Overall drug-likeness (Bickerton 2012) |\n")
        f.write("| Lipinski Ro5 | <=1 violation | Oral bioavailability predictor |\n")
        f.write("| Veber | TPSA<=140, RotB<=10 | Oral bioavailability (rat) |\n")
        f.write("| ESOL logS | >-4 soluble | Aqueous solubility estimate |\n")
        f.write("| Fsp3 | >0.42 preferred | Fraction sp3 carbons (complexity) |\n")
        f.write("| PAINS | 0 alerts | Pan-Assay Interference compounds |\n\n")

        f.write("## References\n\n")
        f.write("- Lipinski et al. Adv Drug Deliv Rev 2001. doi:10.1016/S0169-409X(00)00129-0\n")
        f.write("- Veber et al. J Med Chem 2002. doi:10.1021/jm020017n\n")
        f.write("- Bickerton et al. Nat Chem 2012. doi:10.1038/nchem.1243\n")
        f.write("- Delaney. JCICS 2004. doi:10.1021/ci034243x (ESOL)\n\n")
        f.write("---\n*Generated by PRISM4D ADMET Pipeline v1.0*\n")


def main():
    parser = argparse.ArgumentParser(description="PRISM4D ADMET Prediction")
    parser.add_argument("--sdf-dir", required=True, help="Directory with docked SDF files")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    sdf_dir = Path(args.sdf_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    sdf_files = sorted(sdf_dir.glob("*_rescored.sdf"))
    if not sdf_files:
        sdf_files = sorted(sdf_dir.glob("*.sdf"))

    if not sdf_files:
        print("ERROR: No SDF files found")
        sys.exit(1)

    print("=" * 60)
    print("PRISM4D ADMET & Drug-Likeness Prediction")
    print("=" * 60)

    results = []
    import warnings
    warnings.filterwarnings("ignore")

    for sdf_path in sdf_files:
        lig_name = sdf_path.stem.replace("_rescored", "").replace("_out", "")
        print(f"\n  {lig_name}:")

        suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=True)
        mols = [m for m in suppl if m is not None]
        if not mols:
            print(f"    WARNING: No valid molecules in {sdf_path.name}")
            continue

        mol = mols[0]  # Best pose
        admet = compute_admet(mol, name=lig_name)
        results.append(admet)

        print(f"    MW={admet['molecular_weight']}, cLogP={admet['cLogP']}, "
              f"QED={admet['QED']:.3f}")
        print(f"    Lipinski: {'PASS' if admet['lipinski_pass'] else 'FAIL'}, "
              f"Solubility: {admet['solubility_class']}")

    # Generate reports
    output_md = output_dir / "admet_report.md"
    output_json = output_dir / "admet_results.json"
    generate_admet_report(results, output_md, output_json)
    print(f"\n  Report: {output_md}")
    print(f"  JSON:   {output_json}")
    print("=" * 60)


if __name__ == "__main__":
    main()
