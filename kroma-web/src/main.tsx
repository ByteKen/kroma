import '@fontsource/inter/latin-400.css';
import '@fontsource/inter/latin-500.css';
import '@fontsource/inter/latin-600.css';
import '@fontsource/jetbrains-mono/latin-400.css';
import '@fontsource/jetbrains-mono/latin-500.css';
import '@fontsource/jetbrains-mono/latin-600.css';
import '@fontsource/manrope/latin-600.css';
import '@fontsource/manrope/latin-700.css';
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { App } from './app/App';
import './styles/theme.css';

function isEditableTextTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable) return true;
  if (target instanceof HTMLTextAreaElement) return true;
  if (target instanceof HTMLInputElement) {
    const type = (target.type || 'text').toLowerCase();
    return ['text', 'search', 'url', 'email', 'password', 'number', 'tel'].includes(type);
  }
  return false;
}

// Suppress default WebView2 / WKWebView browser context menu ("Save image as", "Copy image",
// "Reload", "Inspect") across the desktop app while keeping native Copy/Paste inside text inputs
// and when text is explicitly highlighted (e.g. release notes).
document.addEventListener('contextmenu', (event) => {
  if (event.defaultPrevented) return;
  if (isEditableTextTarget(event.target)) return;
  const selectedText = window.getSelection()?.toString().trim() ?? '';
  if (selectedText.length > 0) return;
  event.preventDefault();
});

const rootElement = document.getElementById('root');
if (!rootElement) {
  throw new Error('Root element #root not found');
}

createRoot(rootElement).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
