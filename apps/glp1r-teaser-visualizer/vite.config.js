import fs from "node:fs";
import path from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
var repoRoot = path.resolve(__dirname, "../..");
function prismArtifactServer() {
    return {
        name: "prism-artifact-server",
        configureServer: function (server) {
            server.middlewares.use(function (req, res, next) {
                var _a, _b;
                var url = (_a = req.url) !== null && _a !== void 0 ? _a : "";
                if (!url.startsWith("/campaigns/") && !url.startsWith("/00_registry/")) {
                    next();
                    return;
                }
                var cleanUrl = decodeURIComponent((_b = url.split("?")[0]) !== null && _b !== void 0 ? _b : "");
                var artifactPath = path.resolve(repoRoot, cleanUrl.slice(1));
                if (!artifactPath.startsWith(repoRoot) || !fs.existsSync(artifactPath) || fs.statSync(artifactPath).isDirectory()) {
                    res.statusCode = 404;
                    res.end("artifact not found");
                    return;
                }
                if (artifactPath.endsWith(".parquet")) {
                    res.setHeader("Content-Type", "application/octet-stream");
                }
                else if (artifactPath.endsWith(".json") || artifactPath.endsWith(".jsonl")) {
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
