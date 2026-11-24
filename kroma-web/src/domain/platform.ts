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

const SHARED_SYSTEM_FOLDER_NAMES = new Set([
  'downloads',
  'desktop',
  'documents',
  'temp',
  'tmp',
]);

/**
 * Returns `true` for OS drive roots, user home directories, and shared inbox folders
 * (`Downloads`, `Desktop`, `Documents`, `Temp`) where one-click "Open Parent Folder"
 * would recursively scan unrelated files, screenshots, and archives.
 */
export function isSharedSystemFolder(folderPath: string | null | undefined): boolean {
  if (!folderPath) return false;
  const normalized = folderPath.trim().replace(/[\\/]+/g, '/').replace(/\/+$/, '');
  if (!normalized) return true;
  // Drive roots like `C:` or POSIX `/`
  if (/^[a-z]:$/i.test(normalized)) return true;

  const segments = normalized.split('/').filter(Boolean);
  if (segments.length <= 1) return true;

  const lowerSegments = segments.map((s) => s.toLowerCase());
  const leaf = lowerSegments[lowerSegments.length - 1] ?? '';

  // User home directory: `C:/Users/<name>` or `/Users/<name>` or `/home/<name>`
  if (
    (lowerSegments.length === 3 &&
      /^[a-z]:$/.test(lowerSegments[0]!) &&
      lowerSegments[1] === 'users') ||
    (lowerSegments.length === 2 &&
      (lowerSegments[0] === 'users' || lowerSegments[0] === 'home'))
  ) {
    return true;
  }

  // Standard user inbox folders directly under home (or OneDrive/<Inbox>)
  if (SHARED_SYSTEM_FOLDER_NAMES.has(leaf)) {
    const parent = lowerSegments[lowerSegments.length - 2] ?? '';
    const grandParent = lowerSegments[lowerSegments.length - 3] ?? '';
    if (
      grandParent === 'users' ||
      grandParent === 'home' ||
      parent === 'onedrive' ||
      parent.startsWith('onedrive -') ||
      leaf === 'temp' ||
      leaf === 'tmp'
    ) {
      return true;
    }
  }

  return false;
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
