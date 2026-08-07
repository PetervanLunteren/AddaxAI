import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

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
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      // The job WebSocket lives outside /api, so it needs its own entry.
      // Without it, a dev server with no .env resolves API_BASE_URL to the
      // Vite origin, ws://localhost:5173/ws/jobs/... reaches nothing, and a
      // job never starts: the backend waits for the socket's ready message
      // before it begins work.
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
      },
    },
  },
})
