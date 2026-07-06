import { defineConfig } from 'vite';

// @mujoco/mujoco is an Emscripten ESM module that resolves its .wasm via
// import.meta.url / locateFile. onnxruntime-web resolves its runtime .mjs/.wasm
// the same way. Neither survives esbuild dependency pre-bundling, so both are
// excluded; we pass explicit URLs at runtime (see src/sim/mujoco_sim.ts and
// src/policy/infer.ts).
export default defineConfig({
  optimizeDeps: {
    exclude: ['@mujoco/mujoco', 'onnxruntime-web'],
  },
  build: {
    target: 'es2022',
    // mujoco.wasm alone is ~10 MB; keep the warning threshold honest instead
    // of silencing real regressions in our own code.
    chunkSizeWarningLimit: 1500,
  },
});
