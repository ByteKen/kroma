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
  test: {
    environment: 'node',
    include: ['../tests/web/**/*.test.ts', 'src/**/*.test.ts'],
    passWithNoTests: true,
  },
});
