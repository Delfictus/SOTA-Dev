export const artifactPaths = {
  replayabilityManifest: "/campaigns/glp1r_aleniglipron/M2_Replayability_Manifest.json",
  fragmentInterference: "/campaigns/glp1r_aleniglipron/track_0_manual_emulation/fragment_interference_attribution.parquet",
  teaserSolutions: "/campaigns/glp1r_aleniglipron/track_0_manual_emulation/teaser_solutions.parquet",
  dynamicAlignmentReference: "/campaigns/glp1r_aleniglipron/track_a_generative/dynamic_alignment_reference.json"
} as const;

export type TeaserSolution = {
  solution_rank: bigint | number;
  anchor_id: string;
  source_anchor_id: string;
  canonical_smiles: string;
  sa_score: number;
  pi_clash: number;
  pi_complement: number;
  mapped_atom_count: number;
  signal_mapped_atom_count: number;
  edge_matched_atom_count: number;
  thermally_activated_voxel_count: number;
  scaffold_exit_xyz_json: string;
  exit_vector_json: string;
  aligned_conformer_atoms_json: string;
  liability_edge_id: string;
  liability_edge_label: string;
  condition_id: string;
  projected_durability_improvement?: number;
  anchor_epistemic_class?: string;
  solution_epistemic_class?: string;
};

export type FragmentInterferenceRow = {
  edge_id: string;
  whole_molecule_clash: number;
  whole_molecule_complement: number;
  dominant_fragment: string;
  dominant_fragment_clash: number;
};

export type ManifestSummary = {
  merkle_root?: string;
  unified_merkle_root?: string;
  environment?: Record<string, unknown>;
};

export type ConformerAtom = {
  atom_idx: number;
  atomic_num: number;
  symbol: string;
  x: number;
  y: number;
  z: number;
};
