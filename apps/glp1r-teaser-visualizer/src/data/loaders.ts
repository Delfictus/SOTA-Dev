import initParquet, { readParquet } from "parquet-wasm";
import { tableFromIPC } from "apache-arrow";
import { artifactPaths, type FragmentInterferenceRow, type ManifestSummary, type TeaserSolution } from "./artifacts";

let parquetReady: Promise<void> | null = null;

const ensureParquet = (): Promise<void> => {
  if (parquetReady === null) {
    parquetReady = initParquet().then(() => undefined);
  }
  return parquetReady;
};

export async function loadJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load ${url}: ${response.status}`);
  }
  return (await response.json()) as T;
}

export async function loadParquetRecords<T extends Record<string, unknown>>(url: string): Promise<T[]> {
  await ensureParquet();
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load ${url}: ${response.status}`);
  }
  const parquetBytes = new Uint8Array(await response.arrayBuffer());
  const wasmTable = readParquet(parquetBytes);
  const arrowTable = tableFromIPC(wasmTable.intoIPCStream());
  const rows: T[] = [];
  for (let rowIndex = 0; rowIndex < arrowTable.numRows; rowIndex += 1) {
    const row: Record<string, unknown> = {};
    for (const field of arrowTable.schema.fields) {
      const column = arrowTable.getChild(field.name);
      row[field.name] = column?.get(rowIndex);
    }
    rows.push(row as T);
  }
  return rows;
}

export async function loadVisualizerData(): Promise<{
  manifest: ManifestSummary;
  fragmentInterference: FragmentInterferenceRow[];
  teaserSolutions: TeaserSolution[];
}> {
  const [manifest, fragmentInterference, teaserSolutions] = await Promise.all([
    loadJson<ManifestSummary>(artifactPaths.replayabilityManifest),
    loadParquetRecords<FragmentInterferenceRow>(artifactPaths.fragmentInterference),
    loadParquetRecords<TeaserSolution>(artifactPaths.teaserSolutions)
  ]);
  return { manifest, fragmentInterference, teaserSolutions };
}
