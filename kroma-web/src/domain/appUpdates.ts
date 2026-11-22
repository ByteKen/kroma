export const KROMA_GITHUB_REPO = 'ByteKen/kroma';
export const KROMA_GITHUB_URL = `https://github.com/${KROMA_GITHUB_REPO}`;
export const KROMA_RELEASES_URL = `${KROMA_GITHUB_URL}/releases`;
export const KROMA_RELEASES_API_URL = `https://api.github.com/repos/${KROMA_GITHUB_REPO}/releases?per_page=15`;
export const KROMA_RAW_CONTENT_BASE_URL = `https://raw.githubusercontent.com/${KROMA_GITHUB_REPO}`;
export const KROMA_FALLBACK_VERSION = '0.1.1';
export const DEFAULT_UPDATE_CHECK_TIMEOUT_MS = 4500;

export type ChangelogSource = 'release_body' | 'changelog_file';

export interface AppUpdateCheckResult {
  readonly status: 'up_to_date' | 'update_available' | 'no_releases' | 'error';
  readonly currentVersion: string;
  readonly latestVersion: string | null;
  readonly releaseUrl: string;
  readonly releaseName: string | null;
  readonly publishedAt: string | null;
  readonly changelogMarkdown: string | null;
  readonly changelogSource: ChangelogSource | null;
  readonly message: string;
}

export interface GithubReleaseAssetPayload {
  readonly name?: string;
  readonly browser_download_url?: string;
}

interface GithubReleasePayload {
  readonly tag_name?: string;
  readonly name?: string;
  readonly html_url?: string;
  readonly published_at?: string;
  readonly body?: string | null;
  readonly draft?: boolean;
  readonly prerelease?: boolean;
  readonly assets?: readonly GithubReleaseAssetPayload[];
}

const DESKTOP_SEMVER_TAG_RE = /^v?\d+\.\d+\.\d+(?:[-+].*)?$/i;

/**
 * Returns true only for semver desktop app release tags (`v0.2.0`, `1.0.0`).
 * Excludes auxiliary asset releases such as `models-v1`.
 */
export function isDesktopReleaseTag(tag: string): boolean {
  return DESKTOP_SEMVER_TAG_RE.test(tag.trim());
}

function parseSemverTriplet(raw: string): [number, number, number] {
  const cleaned = raw.trim().replace(/^v/i, '').split(/[-+]/)[0] ?? '';
  const parts = cleaned.split('.').map((p) => {
    const n = Number.parseInt(p, 10);
    return Number.isFinite(n) && n >= 0 ? n : 0;
  });
  return [parts[0] ?? 0, parts[1] ?? 0, parts[2] ?? 0];
}

/**
 * Compares two semver strings (`"0.1.0"`, `"v0.2.1"`).
 * Returns `> 0` when `a > b`, `< 0` when `a < b`, and `0` when equal.
 */
export function compareSemverVersions(a: string, b: string): number {
  const [aMaj, aMin, aPat] = parseSemverTriplet(a);
  const [bMaj, bMin, bPat] = parseSemverTriplet(b);
  if (aMaj !== bMaj) return aMaj - bMaj;
  if (aMin !== bMin) return aMin - bMin;
  return aPat - bPat;
}

/**
 * Extracts the section for `targetVersion` from a Keep-a-Changelog style `CHANGELOG.md`.
 * Falls back to the first `## ` release section or the trimmed file if no exact heading matches.
 */
export function extractChangelogSectionForVersion(
  changelogMd: string,
  targetVersion: string,
): string | null {
  const normalized = changelogMd.replace(/\r\n?/g, '\n').trim();
  if (!normalized) return null;

  const cleanTarget = targetVersion.trim().replace(/^v/i, '');
  const escapedTarget = cleanTarget.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const exactHeadingRe = new RegExp(`^##\\s+\\[?v?${escapedTarget}\\]?(?:\\b|$)`, 'im');
  const lines = normalized.split('\n');

  const sliceFromHeading = (startIdx: number): string => {
    let endIdx = lines.length;
    for (let i = startIdx + 1; i < lines.length; i++) {
      if (/^##\s+/.test(lines[i]!)) {
        endIdx = i;
        break;
      }
    }
    return lines.slice(startIdx, endIdx).join('\n').trim();
  };

  for (let i = 0; i < lines.length; i++) {
    if (exactHeadingRe.test(lines[i]!)) {
      const section = sliceFromHeading(i);
      if (section) return section;
    }
  }

  for (let i = 0; i < lines.length; i++) {
    if (/^##\s+/.test(lines[i]!)) {
      const firstSection = sliceFromHeading(i);
      if (firstSection) return firstSection;
    }
  }

  return normalized;
}

async function fetchWithTimeout(
  url: string,
  init: RequestInit,
  timeoutMs: number,
  fetchFn: typeof fetch,
): Promise<Response> {
  const controller = new AbortController();
  const timer = globalThis.setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetchFn(url, { ...init, signal: controller.signal });
  } finally {
    globalThis.clearTimeout(timer);
  }
}

async function resolveFallbackChangelogMarkdown(
  release: GithubReleasePayload,
  tag: string,
  latestVersion: string,
  timeoutMs: number,
  fetchFn: typeof fetch,
): Promise<string | null> {
  const candidateUrls: string[] = [];

  const changelogAsset = release.assets?.find(
    (asset) =>
      typeof asset.name === 'string' &&
      asset.name.trim().toLowerCase() === 'changelog.md' &&
      typeof asset.browser_download_url === 'string' &&
      asset.browser_download_url.length > 0,
  );
  if (changelogAsset?.browser_download_url) {
    candidateUrls.push(changelogAsset.browser_download_url);
  }

  candidateUrls.push(
    `${KROMA_RAW_CONTENT_BASE_URL}/${encodeURIComponent(tag)}/CHANGELOG.md`,
    `${KROMA_RAW_CONTENT_BASE_URL}/main/CHANGELOG.md`,
  );

  for (const url of candidateUrls) {
    try {
      const res = await fetchWithTimeout(
        url,
        { headers: { Accept: 'text/plain, text/markdown' } },
        timeoutMs,
        fetchFn,
      );
      if (!res.ok) continue;
      const text = await res.text();
      const extracted = extractChangelogSectionForVersion(text, latestVersion);
      if (extracted) {
        return extracted;
      }
    } catch {
      // Try next fallback URL
    }
  }

  return null;
}

function selectLatestDesktopRelease(
  payload: unknown,
): GithubReleasePayload | null {
  const list: readonly GithubReleasePayload[] = Array.isArray(payload)
    ? (payload as readonly GithubReleasePayload[])
    : payload && typeof payload === 'object'
      ? [payload as GithubReleasePayload]
      : [];

  let best: GithubReleasePayload | null = null;
  let bestVersion: string | null = null;

  for (const item of list) {
    if (item.draft || item.prerelease) continue;
    const tag = (item.tag_name ?? '').trim();
    if (!isDesktopReleaseTag(tag)) continue;
    const version = tag.replace(/^v/i, '');
    if (!bestVersion || compareSemverVersions(version, bestVersion) > 0) {
      best = item;
      bestVersion = version;
    }
  }

  return best;
}

export async function checkGithubForAppUpdate(
  currentVersion: string,
  fetchFn: typeof fetch = fetch,
  options?: { readonly timeoutMs?: number },
): Promise<AppUpdateCheckResult> {
  const normalizedCurrent = currentVersion.trim().replace(/^v/i, '') || KROMA_FALLBACK_VERSION;
  const timeoutMs = options?.timeoutMs ?? DEFAULT_UPDATE_CHECK_TIMEOUT_MS;

  try {
    const response = await fetchWithTimeout(
      KROMA_RELEASES_API_URL,
      {
        headers: {
          Accept: 'application/vnd.github+json',
        },
      },
      timeoutMs,
      fetchFn,
    );

    if (response.status === 404) {
      return {
        status: 'no_releases',
        currentVersion: normalizedCurrent,
        latestVersion: null,
        releaseUrl: KROMA_RELEASES_URL,
        releaseName: null,
        publishedAt: null,
        changelogMarkdown: null,
        changelogSource: null,
        message: `Running v${normalizedCurrent} — no published GitHub releases yet.`,
      };
    }

    if (!response.ok) {
      return {
        status: 'error',
        currentVersion: normalizedCurrent,
        latestVersion: null,
        releaseUrl: KROMA_RELEASES_URL,
        releaseName: null,
        publishedAt: null,
        changelogMarkdown: null,
        changelogSource: null,
        message: `Update check returned HTTP ${response.status}.`,
      };
    }

    const payload: unknown = await response.json();
    const release = selectLatestDesktopRelease(payload);

    if (!release) {
      return {
        status: 'no_releases',
        currentVersion: normalizedCurrent,
        latestVersion: null,
        releaseUrl: KROMA_RELEASES_URL,
        releaseName: null,
        publishedAt: null,
        changelogMarkdown: null,
        changelogSource: null,
        message: `Running v${normalizedCurrent} — no published desktop releases yet.`,
      };
    }

    const tag = (release.tag_name ?? '').trim();
    const latest = tag.replace(/^v/i, '');
    const releaseUrl = release.html_url || KROMA_RELEASES_URL;
    const bodyMd = release.body?.trim() || null;

    if (compareSemverVersions(latest, normalizedCurrent) > 0) {
      let changelogMarkdown = bodyMd;
      let changelogSource: ChangelogSource | null = bodyMd ? 'release_body' : null;

      if (!changelogMarkdown) {
        const fallbackMd = await resolveFallbackChangelogMarkdown(
          release,
          tag,
          latest,
          timeoutMs,
          fetchFn,
        );
        if (fallbackMd) {
          changelogMarkdown = fallbackMd;
          changelogSource = 'changelog_file';
        }
      }

      return {
        status: 'update_available',
        currentVersion: normalizedCurrent,
        latestVersion: latest,
        releaseUrl,
        releaseName: release.name?.trim() || `v${latest}`,
        publishedAt: release.published_at ?? null,
        changelogMarkdown,
        changelogSource,
        message: `Update available: v${latest} (you have v${normalizedCurrent}).`,
      };
    }

    return {
      status: 'up_to_date',
      currentVersion: normalizedCurrent,
      latestVersion: latest,
      releaseUrl,
      releaseName: release.name?.trim() || `v${latest}`,
      publishedAt: release.published_at ?? null,
      changelogMarkdown: bodyMd,
      changelogSource: bodyMd ? 'release_body' : null,
      message: `You’re up to date (v${normalizedCurrent}).`,
    };
  } catch {
    return {
      status: 'error',
      currentVersion: normalizedCurrent,
      latestVersion: null,
      releaseUrl: KROMA_RELEASES_URL,
      releaseName: null,
      publishedAt: null,
      changelogMarkdown: null,
      changelogSource: null,
      message: 'Could not reach GitHub (offline or blocked).',
    };
  }
}
