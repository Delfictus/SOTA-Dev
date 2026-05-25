import { Canvas } from "@react-three/fiber";
import { OrbitControls, PerspectiveCamera } from "@react-three/drei";
import { motion } from "framer-motion";
import { useEffect, useMemo, useState } from "react";
import { useTeaserStore, type EpistemicLevel, type SceneIndex } from "./store";
import { loadVisualizerData } from "./data/loaders";
import type { ConformerAtom, FragmentInterferenceRow, ManifestSummary, TeaserSolution } from "./data/artifacts";

type VisualizerData = {
  manifest: ManifestSummary;
  fragmentInterference: FragmentInterferenceRow[];
  teaserSolutions: TeaserSolution[];
};

const atomColor = (symbol: string): string => {
  switch (symbol) {
    case "O":
      return "#e24c4b";
    case "N":
      return "#4c78d8";
    case "F":
      return "#78d278";
    case "P":
      return "#e49b33";
    case "S":
      return "#d6c54c";
    default:
      return "#d9dee8";
  }
};

const parseAtoms = (solution: TeaserSolution | undefined): ConformerAtom[] => {
  if (!solution) {
    return [];
  }
  return JSON.parse(solution.aligned_conformer_atoms_json) as ConformerAtom[];
};

const epistemicLevelByClass: Record<string, EpistemicLevel> = {
  OBSERVED: 1,
  DERIVED: 2,
  INFERRED: 3,
  PROJECTED: 4,
  HYPOTHESIZED: 5
};

const epistemicLevelFor = (value: string | undefined): EpistemicLevel => {
  if (value && value in epistemicLevelByClass) {
    return epistemicLevelByClass[value];
  }
  return 5;
};

function MoleculeMotif({ solution }: { solution: TeaserSolution | undefined }) {
  const atoms = useMemo(() => parseAtoms(solution), [solution]);
  const epistemicFilter = useTeaserStore((state) => state.epistemicFilter);
  if (!solution) {
    return null;
  }
  const exit = JSON.parse(solution.scaffold_exit_xyz_json) as [number, number, number];
  const motifLevel = epistemicLevelFor(solution.solution_epistemic_class ?? solution.anchor_epistemic_class);
  const attenuated = motifLevel > epistemicFilter;
  const atomOpacity = attenuated ? 0.1 : 1.0;
  const cloudOpacity = attenuated ? 0.03 : 0.14;
  return (
    <group>
      <mesh position={exit}>
        <sphereGeometry args={[0.38, 24, 24]} />
        <meshStandardMaterial color="#8a94a6" metalness={0.1} roughness={0.35} transparent={attenuated} opacity={atomOpacity} wireframe={attenuated} />
      </mesh>
      {atoms.map((atom) => (
        <mesh key={atom.atom_idx} position={[atom.x, atom.y, atom.z]}>
          <sphereGeometry args={[0.22, 18, 18]} />
          <meshStandardMaterial color={atomColor(atom.symbol)} roughness={0.28} transparent={attenuated} opacity={atomOpacity} wireframe={attenuated} />
        </mesh>
      ))}
      {atoms.slice(0, 18).map((atom) => (
        <mesh key={`cloud-${atom.atom_idx}`} position={[atom.x, atom.y, atom.z]}>
          <sphereGeometry args={[0.55, 20, 20]} />
          <meshBasicMaterial color="#39d98a" transparent opacity={cloudOpacity} wireframe={attenuated} />
        </mesh>
      ))}
    </group>
  );
}

function SceneCanvas({ solution }: { solution: TeaserSolution | undefined }) {
  const cameraTarget = useTeaserStore((state) => state.cameraTarget);
  return (
    <Canvas>
      <PerspectiveCamera makeDefault position={[cameraTarget[0] + 6, cameraTarget[1] + 5, cameraTarget[2] + 9]} fov={42} />
      <ambientLight intensity={0.7} />
      <directionalLight position={[24, -26, -38]} intensity={1.9} />
      <MoleculeMotif solution={solution} />
      <OrbitControls target={cameraTarget} enableDamping />
    </Canvas>
  );
}

function SceneTabs() {
  const currentScene = useTeaserStore((state) => state.currentScene);
  const setScene = useTeaserStore((state) => state.setScene);
  const labels: Record<SceneIndex, string> = {
    0: "Manifold",
    1: "Lock",
    2: "FRAG-A",
    3: "Interference",
    4: "Solutions"
  };
  return (
    <div className="sceneTabs">
      {([0, 1, 2, 3, 4] as SceneIndex[]).map((scene) => (
        <button key={scene} className={scene === currentScene ? "active" : ""} onClick={() => setScene(scene)}>
          {labels[scene]}
        </button>
      ))}
    </div>
  );
}

function EpistemicControls() {
  const epistemicFilter = useTeaserStore((state) => state.epistemicFilter);
  const setEpistemicFilter = useTeaserStore((state) => state.setEpistemicFilter);
  const options: Array<{ level: EpistemicLevel; label: string }> = [
    { level: 1, label: "Observed Only" },
    { level: 2, label: "Observed + Derived" },
    { level: 5, label: "Full Translational Projection" }
  ];
  return (
    <div className="epistemicControls">
      {options.map((option) => (
        <button
          key={option.level}
          className={option.level === epistemicFilter ? "active" : ""}
          aria-pressed={option.level === epistemicFilter}
          onClick={() => setEpistemicFilter(option.level)}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

function EpistemicWarning({ solution }: { solution: TeaserSolution | undefined }) {
  const currentScene = useTeaserStore((state) => state.currentScene);
  const epistemicFilter = useTeaserStore((state) => state.epistemicFilter);
  const solutionLevel = epistemicLevelFor(solution?.solution_epistemic_class ?? solution?.anchor_epistemic_class);
  if (currentScene !== 4 || solutionLevel < 4 || epistemicFilter < 4) {
    return null;
  }
  return <div className="epistemicWarning">Requires Experimental Falsification.</div>;
}

function SolutionCarousel({ solutions }: { solutions: TeaserSolution[] }) {
  const active = useTeaserStore((state) => state.activeTeaserSolution);
  const setActive = useTeaserStore((state) => state.setActiveTeaserSolution);
  const nextSolution = useTeaserStore((state) => state.nextSolution);
  const previousSolution = useTeaserStore((state) => state.previousSolution);
  const solution = solutions[active];
  return (
    <motion.div className="carousel" initial={{ y: 32, opacity: 0 }} animate={{ y: 0, opacity: 1 }}>
      <div className="carouselHeader">
        <button aria-label="Previous solution" onClick={previousSolution}>‹</button>
        <h2>Zero-Shot Thermodynamic Replacements</h2>
        <button aria-label="Next solution" onClick={nextSolution}>›</button>
      </div>
      {solution ? (
        <div className="solutionMetrics">
          <strong>{solution.anchor_id}</strong>
          <span>Pi complement {Number(solution.pi_complement).toFixed(3)}</span>
          <span>SA {Number(solution.sa_score).toFixed(2)}</span>
          <span>Clash {Number(solution.pi_clash).toFixed(3)}</span>
          <span>{solution.solution_epistemic_class ?? solution.anchor_epistemic_class ?? "HYPOTHESIZED"}</span>
        </div>
      ) : null}
      <div className="solutionButtons">
        {solutions.slice(0, 10).map((item, index) => (
          <button key={item.anchor_id} className={index === active ? "active" : ""} onClick={() => setActive(index)}>
            {Number(item.solution_rank)}
          </button>
        ))}
      </div>
    </motion.div>
  );
}

export function App() {
  const [data, setData] = useState<VisualizerData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const active = useTeaserStore((state) => state.activeTeaserSolution);
  const currentScene = useTeaserStore((state) => state.currentScene);

  useEffect(() => {
    loadVisualizerData().then(setData).catch((reason: unknown) => setError(reason instanceof Error ? reason.message : String(reason)));
  }, []);

  const solution = data?.teaserSolutions[active];

  return (
    <main>
      <SceneTabs />
      <EpistemicControls />
      <section className="viewport">
        <SceneCanvas solution={solution} />
      </section>
      <EpistemicWarning solution={solution} />
      <aside className="proofPanel">
        <p>
          PRISM-DSTW screened 512 calibration motifs and ranked 10 synthetically accessible projected replacements
          against the tensor-defined PHE143 to TYR148 transition-state liability. These hypotheses remain separated
          from observed tensors and require experimental falsification before biological interpretation.
        </p>
        {error ? <p className="error">{error}</p> : null}
        {data ? (
          <p className="lineage">
            Loaded {data.teaserSolutions.length} teaser solutions and {data.fragmentInterference.length} interference rows from campaign artifacts.
          </p>
        ) : (
          <p className="lineage">Loading campaign artifacts...</p>
        )}
      </aside>
      {currentScene === 4 && data ? <SolutionCarousel solutions={data.teaserSolutions} /> : null}
    </main>
  );
}
