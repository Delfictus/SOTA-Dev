#!/usr/bin/env python3
"""
Extract PRISM-Delta binding sites for molecular docking.

Usage:
    python extract_binding_sites.py 6vxx_blind_prediction.json --top 5

Output:
    - binding_site_1.pdb (pocket atoms only)
    - binding_site_1_center.txt (x, y, z for docking grid)
"""

import json
import argparse

def main():
    parser = argparse.ArgumentParser(description='Extract binding sites for docking')
    parser.add_argument('prediction_json', help='PRISM-Delta prediction JSON')
    parser.add_argument('--top', type=int, default=5, help='Number of top sites')
    parser.add_argument('--min-escape', type=float, default=0.5, help='Minimum escape resistance')
    args = parser.parse_args()

    with open(args.prediction_json) as f:
        data = json.load(f)

    sites = data['predicted_sites']
    # Filter by escape resistance and sort by priority
    filtered = [s for s in sites if s['mean_escape_resistance'] >= args.min_escape]
    sorted_sites = sorted(filtered, key=lambda x: x['mean_priority_score'], reverse=True)

    print(f"Found {len(sorted_sites)} sites with escape resistance >= {args.min_escape}")
    print()
    
    for i, site in enumerate(sorted_sites[:args.top]):
        residues = [r['residue_num'] for r in site['residues']]
        center = site['center']
        radius = site['radius']
        
        print(f"=== BINDING SITE {i+1} ===")
        print(f"Residues: {residues}")
        print(f"Center: {center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}")
        print(f"Radius: {radius:.2f} Å")
        print(f"Cryptic Score: {site['mean_cryptic_score']:.3f}")
        print(f"Escape Resistance: {site['mean_escape_resistance']:.3f}")
        print(f"Priority: {site['mean_priority_score']:.3f}")
        print(f"Druggability: {site['druggability_score']:.3f}")
        print()
        
        # PyMOL selection command
        res_str = '+'.join(map(str, residues))
        print(f"PyMOL: select site{i+1}, resi {res_str}")
        print()
        
        # AutoDock Vina grid center
        print(f"AutoDock Vina grid:")
        print(f"  --center_x {center[0]:.2f} --center_y {center[1]:.2f} --center_z {center[2]:.2f}")
        print(f"  --size_x {radius*2+10:.0f} --size_y {radius*2+10:.0f} --size_z {radius*2+10:.0f}")
        print()

if __name__ == '__main__':
    main()

