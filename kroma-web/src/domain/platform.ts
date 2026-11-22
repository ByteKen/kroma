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
