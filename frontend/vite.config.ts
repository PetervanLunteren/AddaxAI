import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// Dev ports, so two checkouts can run side by side against their own
// backend. Set them per worktree in frontend/.env.local (gitignored); the
// defaults are the values of a single checkout. strictPort below makes a
// clash fail: without it Vite moves to the next free port and keeps
// proxying to 8000, which puts this checkout's UI on the other one's
// database with nothing on screen to say so.
const env = loadEnv('development', process.cwd(), 'ADDAXAI_')
const backendPort = env.ADDAXAI_BACKEND_PORT || '8000'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    // Emit .map files so DevTools shows real source locations (file + line)
    // instead of minified offsets. Browsers only fetch the maps when DevTools
    // is open, so end-user runtime cost is zero.
    sourcemap: true,
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: Number(env.ADDAXAI_FRONTEND_PORT || 5173),
    strictPort: true,
    proxy: {
      '/api': {
        target: `http://localhost:${backendPort}`,
        changeOrigin: true,
      },
      // The job WebSocket lives outside /api, so it needs its own entry.
      // Without it, a dev server with no .env resolves API_BASE_URL to the
      // Vite origin, ws://localhost:5173/ws/jobs/... reaches nothing, and a
      // job never starts: the backend waits for the socket's ready message
      // before it begins work.
      '/ws': {
        target: `ws://localhost:${backendPort}`,
        ws: true,
      },
    },
  },
})
