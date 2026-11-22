export interface PlatformEnv {
  readonly platform?: string;
  readonly userAgent?: string;
}

/**
 * Returns true when running on macOS / Apple Silicon / Intel Mac WebView or browser.
 */
export function isMacPlatform(env?: PlatformEnv): boolean {
  const target =
    env ??
    (typeof navigator !== 'undefined'
      ? { platform: navigator.platform, userAgent: navigator.userAgent }
      : undefined);
  if (!target) return false;
  const platform = (target.platform ?? '').toLowerCase();
  const ua = (target.userAgent ?? '').toLowerCase();
  return platform.includes('mac') || ua.includes('macintosh') || ua.includes('mac os x');
}

/**
 * Formats an individual keycap token for the active OS (`Ctrl` -> `⌘`, `Alt` -> `⌥` on macOS).
 */
export function formatShortcutKey(key: string, isMac = isMacPlatform()): string {
  if (!isMac) return key;
  if (key === 'Ctrl') return '⌘';
  if (key === 'Alt') return '⌥';
  return key;
}

/**
 * Formats a compact shortcut label such as `Ctrl+K`, `Ctrl+,`, or `Ctrl+Z` for the active OS.
 */
export function formatShortcutCombo(combo: string, isMac = isMacPlatform()): string {
  if (!isMac) return combo;
  return combo.replace(/\bCtrl\+/gi, '⌘').replace(/\bAlt\+/gi, '⌥');
}

/**
 * Opens an external `http://` or `https://` URL in the user's default system browser.
 * In Tauri v2 Desktop, WebView2 / WKWebView blocks `<a target="_blank">` popups unless routed
 * through the native OS protocol handler (`open_external_url`).
 */
export async function openExternalUrl(url: string): Promise<void> {
  const trimmed = url.trim();
  if (!trimmed.startsWith('https://') && !trimmed.startsWith('http://')) {
    return;
  }
  if (typeof window !== 'undefined') {
    const tauriInternals = (
      window as Window & {
        __TAURI_INTERNALS__?: {
          invoke?: (cmd: string, args?: Record<string, unknown>) => Promise<unknown>;
        };
      }
    ).__TAURI_INTERNALS__;
    if (typeof tauriInternals?.invoke === 'function') {
      try {
        await tauriInternals.invoke('open_external_url', { url: trimmed });
        return;
      } catch {
        // Fall through to window.open fallback
      }
    }
    window.open(trimmed, '_blank', 'noopener,noreferrer');
  }
}
