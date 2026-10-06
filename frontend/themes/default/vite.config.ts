import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { offlineBundle } from '@knowva/core/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), offlineBundle()],
  build: {
    // The backend serves frontend/dist/<theme>, the theme the deployed
    // profile names. Each theme empties only its own directory; the root
    // build clears the whole of frontend/dist first.
    outDir: '../../dist/default',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/api': 'http://localhost:8080',
    },
  },
})
