import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The kiosk talks to the backend at /api/v1 (same origin in production, proxied
// by the frontend's Caddy). In dev, proxy it to a local/remote backend.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.VITE_DEV_API ?? "http://127.0.0.1:8800",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: false,
  },
});
