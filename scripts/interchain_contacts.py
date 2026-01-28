#!/usr/bin/env python3
"""
PRISM4D Inter-Chain Contact Analyzer

Analyzes inter-chain contacts to determine if a structure should use
multi-chain processing (independent chains) or whole-structure processing
(tightly coupled chains).

Contact Types Detected:
1. Close contacts (<4.0 Å between heavy atoms)
2. Potential hydrogen bonds (<3.5 Å donor-acceptor)
3. Disulfide bonds (S-S <2.5 Å)
4. Salt bridges (charged residue pairs <4.0 Å)

Routing Decision:
- Low contact density → Multi-chain processing (chains are independent)
- High contact density → Whole-structure processing (chains are coupled)

Usage:
    python interchain_contacts.py input.pdb
    python interchain_contacts.py input.pdb --threshold 0.5
"""

import os
import sys
import math
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set, NamedTuple
from collections import defaultdict
from dataclasses import dataclass


# Distance thresholds (Å)
CLOSE_CONTACT_CUTOFF = 4.0
HBOND_CUTOFF = 3.5
DISULFIDE_CUTOFF = 2.5
SALT_BRIDGE_CUTOFF = 4.0

# Atoms that can participate in hydrogen bonds
HBOND_DONORS = {'N', 'O', 'S'}  # When bonded to H
HBOND_ACCEPTORS = {'O', 'N', 'S'}

# Charged residues
POSITIVE_RESIDUES = {'ARG', 'LYS', 'HIS'}
NEGATIVE_RESIDUES = {'ASP', 'GLU'}

# Contact density thresholds for routing decision
# Contacts per interface residue pair
# Calibrated from empirical testing:
#   5IRE (Zika): density=1.14, benefits from MULTICHAIN
#   1HXY (HIV):  density=3.04, needs WHOLE-structure
LOW_CONTACT_THRESHOLD = 1.5   # Below this → multi-chain OK
HIGH_CONTACT_THRESHOLD = 2.0  # Above this → definitely whole-structure


@dataclass
class Atom:
    """Represents a PDB atom."""
    serial: int
    name: str
    res_name: str
    chain_id: str
    res_seq: int
    x: float
    y: float
    z: float
    element: str

    def distance_to(self, other: 'Atom') -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)


@dataclass
class Contact:
    """Represents an inter-chain contact."""
    atom1: Atom
    atom2: Atom
    distance: float
    contact_type: str  # 'close', 'hbond', 'disulfide', 'salt_bridge'


@dataclass
class ChainPairAnalysis:
    """Analysis of contacts between two chains."""
    chain1: str
    chain2: str
    n_close_contacts: int
    n_hbonds: int
    n_disulfides: int
    n_salt_bridges: int
    interface_residues_chain1: Set[int]
    interface_residues_chain2: Set[int]
    contact_density: float  # contacts per interface residue


@dataclass
class ContactAnalysisResult:
    """Complete inter-chain contact analysis result."""
    pdb_path: str
    n_chains: int
    chains: List[str]
    chain_sizes: Dict[str, int]  # atoms per chain
    chain_pairs: List[ChainPairAnalysis]
    total_contacts: int
    total_hbonds: int
    total_disulfides: int
    total_salt_bridges: int
    overall_contact_density: float
    recommendation: str  # 'multichain', 'whole', 'standard'
    confidence: float  # 0-1


def parse_pdb(pdb_path: str) -> List[Atom]:
    """Parse a PDB file and return list of atoms."""
    atoms = []

    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                try:
                    serial = int(line[6:11].strip())
                    name = line[12:16].strip()
                    res_name = line[17:20].strip()
                    chain_id = line[21:22].strip() or 'A'
                    res_seq = int(line[22:26].strip())
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())

                    # Get element from columns 77-78 or infer from atom name
                    if len(line) >= 78:
                        element = line[76:78].strip()
                    else:
                        element = name[0] if name else 'C'

                    atoms.append(Atom(
                        serial=serial,
                        name=name,
                        res_name=res_name,
                        chain_id=chain_id,
                        res_seq=res_seq,
                        x=x, y=y, z=z,
                        element=element
                    ))
                except (ValueError, IndexError):
                    continue

    return atoms


def is_heavy_atom(atom: Atom) -> bool:
    """Check if atom is a heavy (non-hydrogen) atom."""
    return atom.element not in ('H', 'D')


def find_interchain_contacts(atoms: List[Atom]) -> Dict[Tuple[str, str], List[Contact]]:
    """Find all inter-chain contacts."""
    # Group atoms by chain
    chains = defaultdict(list)
    for atom in atoms:
        if is_heavy_atom(atom):
            chains[atom.chain_id].append(atom)

    chain_ids = sorted(chains.keys())
    contacts_by_pair = {}

    # Check all chain pairs
    for i, chain1 in enumerate(chain_ids):
        for chain2 in chain_ids[i+1:]:
            contacts = []
            atoms1 = chains[chain1]
            atoms2 = chains[chain2]

            for a1 in atoms1:
                for a2 in atoms2:
                    dist = a1.distance_to(a2)

                    if dist > CLOSE_CONTACT_CUTOFF:
                        continue

                    # Classify contact type
                    contact_type = 'close'

                    # Check for disulfide bond
                    if (a1.element == 'S' and a2.element == 'S' and
                        a1.name == 'SG' and a2.name == 'SG' and
                        dist < DISULFIDE_CUTOFF):
                        contact_type = 'disulfide'

                    # Check for salt bridge
                    elif (a1.res_name in POSITIVE_RESIDUES and a2.res_name in NEGATIVE_RESIDUES) or \
                         (a1.res_name in NEGATIVE_RESIDUES and a2.res_name in POSITIVE_RESIDUES):
                        if dist < SALT_BRIDGE_CUTOFF:
                            # Check if it's a charged atom (simplified)
                            if (a1.name in ('NZ', 'NH1', 'NH2', 'NE', 'ND1', 'NE2', 'OD1', 'OD2', 'OE1', 'OE2') and
                                a2.name in ('NZ', 'NH1', 'NH2', 'NE', 'ND1', 'NE2', 'OD1', 'OD2', 'OE1', 'OE2')):
                                contact_type = 'salt_bridge'

                    # Check for hydrogen bond
                    elif dist < HBOND_CUTOFF:
                        if ((a1.element in HBOND_DONORS and a2.element in HBOND_ACCEPTORS) or
                            (a1.element in HBOND_ACCEPTORS and a2.element in HBOND_DONORS)):
                            # Backbone or sidechain H-bond candidates
                            if a1.name in ('N', 'O', 'OG', 'OG1', 'OD1', 'OD2', 'OE1', 'OE2', 'NE', 'NH1', 'NH2', 'NZ', 'ND1', 'ND2', 'NE1', 'NE2') or \
                               a2.name in ('N', 'O', 'OG', 'OG1', 'OD1', 'OD2', 'OE1', 'OE2', 'NE', 'NH1', 'NH2', 'NZ', 'ND1', 'ND2', 'NE1', 'NE2'):
                                contact_type = 'hbond'

                    contacts.append(Contact(
                        atom1=a1,
                        atom2=a2,
                        distance=dist,
                        contact_type=contact_type
                    ))

            contacts_by_pair[(chain1, chain2)] = contacts

    return contacts_by_pair


def analyze_chain_pair(chain1: str, chain2: str, contacts: List[Contact]) -> ChainPairAnalysis:
    """Analyze contacts between a pair of chains."""
    interface_res1 = set()
    interface_res2 = set()
    n_close = 0
    n_hbonds = 0
    n_disulfides = 0
    n_salt_bridges = 0

    for contact in contacts:
        interface_res1.add(contact.atom1.res_seq)
        interface_res2.add(contact.atom2.res_seq)

        if contact.contact_type == 'close':
            n_close += 1
        elif contact.contact_type == 'hbond':
            n_hbonds += 1
        elif contact.contact_type == 'disulfide':
            n_disulfides += 1
        elif contact.contact_type == 'salt_bridge':
            n_salt_bridges += 1

    # Calculate contact density
    n_interface_res = len(interface_res1) + len(interface_res2)
    total_contacts = n_close + n_hbonds + n_disulfides + n_salt_bridges
    contact_density = total_contacts / max(n_interface_res, 1)

    return ChainPairAnalysis(
        chain1=chain1,
        chain2=chain2,
        n_close_contacts=n_close,
        n_hbonds=n_hbonds,
        n_disulfides=n_disulfides,
        n_salt_bridges=n_salt_bridges,
        interface_residues_chain1=interface_res1,
        interface_residues_chain2=interface_res2,
        contact_density=contact_density
    )


def make_routing_decision(chain_pairs: List[ChainPairAnalysis], n_chains: int) -> Tuple[str, float]:
    """
    Decide whether to use multi-chain or whole-structure processing.

    Decision based on empirical testing:
    - 5IRE (Zika, 6 chains): density=1.14, benefits from MULTICHAIN
    - 1HXY (HIV, 4 chains):  density=3.04, needs WHOLE-structure

    Returns (recommendation, confidence)
    """
    if n_chains <= 2:
        return 'standard', 1.0

    if not chain_pairs:
        return 'multichain', 0.9

    # Calculate overall metrics
    n_pairs = len(chain_pairs)
    total_contacts = sum(p.n_close_contacts + p.n_hbonds + p.n_disulfides + p.n_salt_bridges
                        for p in chain_pairs)
    total_disulfides = sum(p.n_disulfides for p in chain_pairs)
    total_hbonds = sum(p.n_hbonds for p in chain_pairs)
    total_salt_bridges = sum(p.n_salt_bridges for p in chain_pairs)

    avg_contact_density = sum(p.contact_density for p in chain_pairs) / n_pairs
    max_contact_density = max(p.contact_density for p in chain_pairs)

    # Per-pair averages (more robust for different chain counts)
    hbonds_per_pair = total_hbonds / n_pairs
    salt_bridges_per_pair = total_salt_bridges / n_pairs

    # Decision logic (calibrated from empirical testing)

    # 1. Disulfide bonds are very strong indicators of coupled chains
    if total_disulfides > 0:
        return 'whole', 0.95

    # 2. Many salt bridges per chain pair suggest strong coupling
    #    1HXY: 13 salt bridges / 6 pairs = 2.17 per pair
    #    5IRE: 5 salt bridges / 15 pairs = 0.33 per pair
    if salt_bridges_per_pair > 1.0:
        return 'whole', 0.85

    # 3. High H-bond density per pair suggests coupled chains
    #    1HXY: 99 H-bonds / 6 pairs = 16.5 per pair
    #    5IRE: 76 H-bonds / 15 pairs = 5.1 per pair
    if hbonds_per_pair > 10.0:
        return 'whole', 0.80

    # 4. Check overall contact density
    #    Key threshold: 5IRE has 1.14, 1HXY has 3.04
    #    Use 2.0 as cutoff
    if avg_contact_density > HIGH_CONTACT_THRESHOLD:
        confidence = min(0.9, 0.6 + (avg_contact_density - HIGH_CONTACT_THRESHOLD) * 0.2)
        return 'whole', confidence

    if avg_contact_density < LOW_CONTACT_THRESHOLD:
        confidence = min(0.9, 0.7 + (LOW_CONTACT_THRESHOLD - avg_contact_density) * 0.2)
        return 'multichain', confidence

    # 5. Middle ground - lean towards multichain for borderline cases
    #    since incorrect multichain processing is recoverable (just reprocess)
    if avg_contact_density < (LOW_CONTACT_THRESHOLD + HIGH_CONTACT_THRESHOLD) / 2:
        return 'multichain', 0.65
    else:
        return 'whole', 0.55


def analyze_structure(pdb_path: str, verbose: bool = True) -> ContactAnalysisResult:
    """
    Perform complete inter-chain contact analysis.

    Returns ContactAnalysisResult with recommendation.
    """
    atoms = parse_pdb(pdb_path)

    # Group by chain
    chain_atoms = defaultdict(list)
    for atom in atoms:
        chain_atoms[atom.chain_id].append(atom)

    chains = sorted(chain_atoms.keys())
    n_chains = len(chains)
    chain_sizes = {c: len(chain_atoms[c]) for c in chains}

    if verbose:
        print(f"\n{'='*60}")
        print(f"Inter-Chain Contact Analysis: {Path(pdb_path).name}")
        print(f"{'='*60}")
        print(f"Chains: {n_chains} ({', '.join(chains)})")
        for c in chains:
            print(f"  Chain {c}: {chain_sizes[c]} atoms")

    # Find contacts
    contacts_by_pair = find_interchain_contacts(atoms)

    # Analyze each pair
    chain_pair_analyses = []
    for (c1, c2), contacts in contacts_by_pair.items():
        analysis = analyze_chain_pair(c1, c2, contacts)
        chain_pair_analyses.append(analysis)

    # Totals
    total_contacts = sum(len(c) for c in contacts_by_pair.values())
    total_hbonds = sum(p.n_hbonds for p in chain_pair_analyses)
    total_disulfides = sum(p.n_disulfides for p in chain_pair_analyses)
    total_salt_bridges = sum(p.n_salt_bridges for p in chain_pair_analyses)

    overall_density = sum(p.contact_density for p in chain_pair_analyses) / max(len(chain_pair_analyses), 1)

    # Make routing decision
    recommendation, confidence = make_routing_decision(chain_pair_analyses, n_chains)

    if verbose:
        print(f"\nInter-Chain Contacts:")
        print(f"  Total close contacts: {total_contacts}")
        print(f"  Hydrogen bonds: {total_hbonds}")
        print(f"  Disulfide bonds: {total_disulfides}")
        print(f"  Salt bridges: {total_salt_bridges}")
        print(f"  Avg contact density: {overall_density:.3f}")

        if chain_pair_analyses:
            print(f"\nPer-Chain-Pair Analysis:")
            for p in chain_pair_analyses:
                print(f"  {p.chain1}-{p.chain2}: {p.n_close_contacts} contacts, "
                      f"{p.n_hbonds} H-bonds, {p.n_disulfides} S-S, "
                      f"density={p.contact_density:.3f}")

        print(f"\n{'='*60}")
        rec_symbol = {'multichain': '🔀', 'whole': '📦', 'standard': '✨'}[recommendation]
        print(f"RECOMMENDATION: {rec_symbol} {recommendation.upper()} processing")
        print(f"Confidence: {confidence:.0%}")
        print(f"{'='*60}\n")

    return ContactAnalysisResult(
        pdb_path=pdb_path,
        n_chains=n_chains,
        chains=chains,
        chain_sizes=chain_sizes,
        chain_pairs=chain_pair_analyses,
        total_contacts=total_contacts,
        total_hbonds=total_hbonds,
        total_disulfides=total_disulfides,
        total_salt_bridges=total_salt_bridges,
        overall_contact_density=overall_density,
        recommendation=recommendation,
        confidence=confidence
    )


def main():
    parser = argparse.ArgumentParser(
        description='Analyze inter-chain contacts to determine processing strategy'
    )
    parser.add_argument('pdb', help='Input PDB file')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet output')
    parser.add_argument('--json', '-j', action='store_true', help='Output as JSON')

    args = parser.parse_args()

    if not Path(args.pdb).exists():
        print(f"ERROR: File not found: {args.pdb}", file=sys.stderr)
        return 1

    result = analyze_structure(args.pdb, verbose=not args.quiet and not args.json)

    if args.json:
        import json
        output = {
            'pdb': result.pdb_path,
            'n_chains': result.n_chains,
            'chains': result.chains,
            'chain_sizes': result.chain_sizes,
            'total_contacts': result.total_contacts,
            'total_hbonds': result.total_hbonds,
            'total_disulfides': result.total_disulfides,
            'total_salt_bridges': result.total_salt_bridges,
            'overall_contact_density': result.overall_contact_density,
            'recommendation': result.recommendation,
            'confidence': result.confidence
        }
        print(json.dumps(output, indent=2))

    return 0


if __name__ == '__main__':
    sys.exit(main())
