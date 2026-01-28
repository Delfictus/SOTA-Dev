#!/usr/bin/env python3
"""
Convert PRISM4D JSON predictions to PDB with scores mapped to B-factors.
This preserves full atomic fidelity from the original PDB.
"""

import json
import sys
import os
import urllib.request

def fetch_pdb(pdb_id, output_path):
    """Download PDB file from RCSB."""
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    print(f"Fetching {pdb_id} from RCSB...")
    urllib.request.urlretrieve(url, output_path)
    print(f"Saved to {output_path}")

def load_predictions(json_path):
    """Load PRISM4D predictions from JSON."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Build lookup: (chain_id, residue_num) -> scores
    residue_scores = {}
    for res in data.get('residue_predictions', []):
        key = (res['chain_id'], res['residue_num'])
        residue_scores[key] = {
            'cryptic_score': res.get('cryptic_score', 0.0),
            'escape_resistance': res.get('escape_resistance', 0.0),
            'priority_score': res.get('priority_score', 0.0),
            'rmsf': res.get('rmsf', 0.0),
            'burial_fraction': res.get('burial_fraction', 0.0),
        }

    return data, residue_scores

def map_scores_to_pdb(pdb_path, output_path, residue_scores, score_type='cryptic_score'):
    """
    Read PDB, replace B-factors with scores, write new PDB.
    Preserves all atomic coordinates and structure exactly.
    """
    lines_out = []
    modified_count = 0

    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                # PDB format columns:
                # 1-6: Record name, 7-11: Atom serial, 13-16: Atom name
                # 17: Alt loc, 18-20: Residue name, 22: Chain ID
                # 23-26: Residue seq number, 27: iCode
                # 31-38: X, 39-46: Y, 47-54: Z
                # 55-60: Occupancy, 61-66: B-factor

                chain_id = line[21].strip()
                try:
                    res_num = int(line[22:26].strip())
                except ValueError:
                    lines_out.append(line)
                    continue

                key = (chain_id, res_num)
                if key in residue_scores:
                    score = residue_scores[key].get(score_type, 0.0)
                    # Scale score to 0-100 range for B-factor visualization
                    b_factor = score * 100.0
                    # Replace B-factor field (columns 61-66)
                    new_line = line[:60] + f"{b_factor:6.2f}" + line[66:]
                    lines_out.append(new_line)
                    modified_count += 1
                else:
                    # Keep original B-factor for residues not in predictions
                    lines_out.append(line)
            else:
                lines_out.append(line)

    with open(output_path, 'w') as f:
        f.writelines(lines_out)

    print(f"Modified {modified_count} atoms with {score_type}")
    return modified_count

def main():
    if len(sys.argv) < 2:
        print("Usage: python json_to_pdb.py <predictions.json> [score_type]")
        print("Score types: cryptic_score, escape_resistance, priority_score, rmsf, burial_fraction")
        sys.exit(1)

    json_path = sys.argv[1]
    score_type = sys.argv[2] if len(sys.argv) > 2 else 'cryptic_score'

    # Load predictions
    data, residue_scores = load_predictions(json_path)
    pdb_id = data.get('pdb_id', 'UNKNOWN')

    print(f"Loaded {len(residue_scores)} residue predictions for {pdb_id}")

    # Output paths
    base_dir = os.path.dirname(json_path)
    pdb_original = os.path.join(base_dir, f"{pdb_id.lower()}_original.pdb")
    pdb_scored = os.path.join(base_dir, f"{pdb_id.lower()}_{score_type}.pdb")

    # Fetch original PDB if not exists
    if not os.path.exists(pdb_original):
        fetch_pdb(pdb_id, pdb_original)

    # Map scores to B-factors
    map_scores_to_pdb(pdb_original, pdb_scored, residue_scores, score_type)

    print(f"\nOutput: {pdb_scored}")
    print(f"\nIn PyMOL, visualize with:")
    print(f"  load {pdb_scored}")
    print(f"  spectrum b, blue_white_red, minimum=0, maximum=100")
    print(f"  show cartoon")

if __name__ == '__main__':
    main()
