import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { offlineBundle } from '@knowva/core/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), offlineBundle()],
  build: {
    // The backend and the Docker image serve a single bundle from
    // frontend/dist, so the default theme still writes there.
    outDir: '../../dist',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/api': 'http://localhost:8080',
    },
  },
})
