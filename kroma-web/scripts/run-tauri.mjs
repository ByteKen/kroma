import { spawn } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = dirname(fileURLToPath(import.meta.url));
const webDir = resolve(scriptDir, '..');
const tauriProjectDir = resolve(webDir, '..', 'kroma-tau');
const args = process.argv.slice(2);

const isWindows = process.platform === 'win32';
const command = isWindows
  ? process.env.ComSpec ?? 'cmd.exe'
  : resolve(webDir, 'node_modules', '.bin', 'tauri');
const commandArgs = isWindows
  ? ['/d', '/s', '/c', resolve(webDir, 'node_modules', '.bin', 'tauri.cmd'), ...args]
  : args;

const child = spawn(command, commandArgs, {
  cwd: tauriProjectDir,
  stdio: 'inherit',
});

child.on('exit', (code) => {
  process.exit(code ?? 0);
});
