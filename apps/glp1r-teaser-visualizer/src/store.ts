import { create } from "zustand";

export type SceneIndex = 0 | 1 | 2 | 3 | 4;
export type CameraTarget = [number, number, number];
export type EpistemicLevel = 1 | 2 | 3 | 4 | 5;

type TeaserVisualizerState = {
  currentScene: SceneIndex;
  cameraTarget: CameraTarget;
  activeTeaserSolution: number;
  epistemicFilter: EpistemicLevel;
  setScene: (scene: SceneIndex) => void;
  setCameraTarget: (target: CameraTarget) => void;
  setActiveTeaserSolution: (index: number) => void;
  setEpistemicFilter: (level: EpistemicLevel) => void;
  nextSolution: () => void;
  previousSolution: () => void;
};

const sceneTargets: Record<SceneIndex, CameraTarget> = {
  0: [17.7, -37.6, -52.4],
  1: [21.0, -36.1, -56.7],
  2: [19.8, -35.1, -53.2],
  3: [18.4, -38.4, -53.8],
  4: [20.3, -34.0, -52.5]
};

const clampSolution = (index: number): number => Math.max(0, Math.min(9, Math.trunc(index)));

export const useTeaserStore = create<TeaserVisualizerState>((set) => ({
  currentScene: 0,
  cameraTarget: sceneTargets[0],
  activeTeaserSolution: 0,
  epistemicFilter: 5,
  setScene: (scene) =>
    set({
      currentScene: scene,
      cameraTarget: sceneTargets[scene]
    }),
  setCameraTarget: (target) => set({ cameraTarget: target }),
  setActiveTeaserSolution: (index) => set({ activeTeaserSolution: clampSolution(index) }),
  setEpistemicFilter: (level) => set({ epistemicFilter: level }),
  nextSolution: () =>
    set((state) => ({
      activeTeaserSolution: (state.activeTeaserSolution + 1) % 10
    })),
  previousSolution: () =>
    set((state) => ({
      activeTeaserSolution: (state.activeTeaserSolution + 9) % 10
    }))
}));
