// @ts-check
import { defineConfig } from "astro/config";
import preact from "@astrojs/preact";

// Static-first: output: 'static' is Astro's default, stated here for the record.
// Islands (preact) hydrate ONLY where a client:* directive is used — every page
// is zero-JS by default, which is what makes chapters readable with JS disabled.
export default defineConfig({
  output: "static",
  integrations: [preact()],
  // Deterministic, hash-free asset names keep the spike's dist/ easy to inspect.
  build: { assets: "_assets" },
  // P2 SPIKE: the ch1.1 concept-toy island imports the playground's MuJoCo-WASM +
  // ONNX primitives directly (monorepo source sharing — see the SPIKE report and
  // decision 011 proposal). @mujoco/mujoco and onnxruntime-web are Emscripten ESM
  // modules that resolve their .wasm via import.meta.url / locateFile; that does
  // NOT survive esbuild dependency pre-bundling, so both are excluded here exactly
  // as playground/vite.config.ts does. The island passes explicit Vite-resolved
  // ?url paths at runtime (playground/src/sim/mujoco_sim.ts + policy/infer.ts).
  vite: {
    optimizeDeps: {
      exclude: ["@mujoco/mujoco", "onnxruntime-web"],
    },
  },
});
