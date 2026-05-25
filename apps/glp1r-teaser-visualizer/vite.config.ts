import fs from "node:fs";
import path from "node:path";
import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";

const repoRoot = path.resolve(__dirname, "../..");

function prismArtifactServer(): Plugin {
  return {
    name: "prism-artifact-server",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const url = req.url ?? "";
        if (!url.startsWith("/campaigns/") && !url.startsWith("/00_registry/")) {
          next();
          return;
        }
        const cleanUrl = decodeURIComponent(url.split("?")[0] ?? "");
        const artifactPath = path.resolve(repoRoot, cleanUrl.slice(1));
        if (!artifactPath.startsWith(repoRoot) || !fs.existsSync(artifactPath) || fs.statSync(artifactPath).isDirectory()) {
          res.statusCode = 404;
          res.end("artifact not found");
          return;
        }
        if (artifactPath.endsWith(".parquet")) {
          res.setHeader("Content-Type", "application/octet-stream");
        } else if (artifactPath.endsWith(".json") || artifactPath.endsWith(".jsonl")) {
          res.setHeader("Content-Type", "application/json");
        }
        fs.createReadStream(artifactPath).pipe(res);
      });
    }
  };
}

export default defineConfig({
  plugins: [react(), prismArtifactServer()],
  server: {
    fs: {
      allow: [repoRoot]
    }
  }
});
