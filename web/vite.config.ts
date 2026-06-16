import { defineConfig } from 'vite';
import wasm from 'vite-plugin-wasm';
import topLevelAwait from 'vite-plugin-top-level-await';
import { readFileSync } from 'fs';
import { execSync } from 'child_process';

const pkg = JSON.parse(readFileSync('./package.json', 'utf-8'));

// Short git commit hash, captured at build time. Falls back to 'dev' when git
// is unavailable (e.g. building from a tarball without history).
const gitCommit = (() => {
    try {
        return execSync('git rev-parse --short HEAD').toString().trim();
    } catch {
        return 'dev';
    }
})();

const buildDate = new Date().toISOString().slice(0, 10);

export default defineConfig({
    plugins: [wasm(), topLevelAwait()],
    define: {
        __APP_VERSION__: JSON.stringify(pkg.version),
        __GIT_COMMIT__: JSON.stringify(gitCommit),
        __BUILD_DATE__: JSON.stringify(buildDate),
    },
    build: {
        target: 'esnext',
    },
    base: '/2048/', // GitHub Pages subdirectory
});
