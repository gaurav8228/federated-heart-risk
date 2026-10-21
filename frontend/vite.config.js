// Vite config: React plugin + the default dev server (http://localhost:5173).
// The backend is NOT proxied - the app calls http://localhost:8000 directly
// (see src/api.js), which is allowed by the CORS settings in backend/main.py.
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
});
