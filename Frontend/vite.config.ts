import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'

// https://vite.dev/config/
export default defineConfig({
  // Served from https://<user>.github.io/Sail_General/ on GitHub Pages.
  base: '/Sail_General/',
  plugins: [react()],
  server: {
    port: 5174,
    strictPort: true,
  },
})
