import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  plugins: [react(), tailwindcss()],
  // Tauri v2 expects a fixed dev server port and un-cleared terminal output
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
    fs: {
      allow: ['..'],
    },
    watch: {
      ignored: ['**/internal/**', '**/kroma-tau/**', '**/target/**'],
    },
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (!id.includes('node_modules')) return undefined;
          if (
            id.includes('/node_modules/react/') ||
            id.includes('/node_modules/react-dom/') ||
            id.includes('/node_modules/scheduler/') ||
            id.includes('\\node_modules\\react\\') ||
            id.includes('\\node_modules\\react-dom\\') ||
            id.includes('\\node_modules\\scheduler\\')
          ) {
            return 'vendor-react';
          }
          if (
            id.includes('framer-motion') ||
            id.includes('motion-dom') ||
            id.includes('motion-utils')
          ) {
            return 'vendor-motion';
          }
          if (id.includes('@tauri-apps')) {
            return 'vendor-tauri';
          }
          return undefined;
        },
      },
    },
  },
  test: {
    environment: 'node',
    include: ['../tests/web/**/*.test.ts', 'src/**/*.test.ts'],
    passWithNoTests: true,
  },
});
