import { defineConfig } from "vite";

/**
 * The API and the practice socket are proxied, so the page talks to its own
 * origin and CORS never enters the picture during development. In production
 * Nginx plays this role (CLAUDE.md 14).
 */
export default defineConfig({
  server: {
    port: 5173,
    proxy: {
      // Keys match by prefix, so each ends where its route does: a bare "/me"
      // also caught "/mediapipe/..." and sent the hand model to the API (404).
      // Keys starting with "^" are regular expressions.
      "/auth/": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/me/": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "^/letters(/|$)": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "^/health$": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/ws/": { target: "ws://127.0.0.1:8000", ws: true },
    },
  },
});
