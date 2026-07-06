/// <reference types="vite/client" />

// Asset-URL imports used to hand Emscripten/ort explicit .wasm locations.
declare module '@mujoco/mujoco/mujoco.wasm?url' {
  const url: string;
  export default url;
}
declare module 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url' {
  const url: string;
  export default url;
}
