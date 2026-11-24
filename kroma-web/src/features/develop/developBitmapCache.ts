/**
 * Bounded LRU cache of pre-decoded, downscaled `ImageBitmap` frames for 8-bit Studio Develop
 * sources (JPEG, PNG, WebP).
 *
 * Why this exists:
 * - Standard 24–60 MP camera JPEGs render directly from disk (`asset://...`). Without downscaling
 *   and caching, every filmstrip click decodes a 96–240 MB RGBA bitmap and copies it across PCIe
 *   to GPU VRAM (and 2× in Split / Y|Y compare modes).
 * - Downscaling during `createImageBitmap` to `maxDim` (matching Kroma's Studio Preview resolution)
 *   shrinks each upload by ~8–15×, coalesces duplicate Split/Y|Y decodes, and lets neighbor
 *   pre-decoding hit synchronously in <3 ms on frame switch.
 */
export interface CachedDevelopBitmap {
  readonly bitmap: ImageBitmap;
  readonly width: number;
  readonly height: number;
  readonly naturalWidth: number;
  readonly naturalHeight: number;
}

const MAX_CACHED_BITMAPS = 16;
const MAX_HI_RES_BITMAPS = 2;

const bitmapLru = new Map<string, CachedDevelopBitmap>();
const hiResBitmapLru = new Map<string, CachedDevelopBitmap>();
const inFlightLoads = new Map<string, Promise<CachedDevelopBitmap | null>>();

function isHiResMaxDim(maxDim: number): boolean {
  return maxDim === 0 || maxDim > 2560;
}

function makeCacheKey(src: string, maxDim: number): string {
  return `${Math.max(0, Math.round(maxDim))}|${src}`;
}

/**
 * Computes aspect-preserving target bitmap dimensions capped at `maxDim` (`0` = no resize).
 */
export function computeTargetBitmapSize(
  naturalWidth: number,
  naturalHeight: number,
  maxDim: number,
): { readonly width: number; readonly height: number; readonly needsResize: boolean } {
  if (naturalWidth <= 0 || naturalHeight <= 0) {
    return { width: 1, height: 1, needsResize: false };
  }
  const longest = Math.max(naturalWidth, naturalHeight);
  if (maxDim <= 0 || longest <= maxDim) {
    return { width: naturalWidth, height: naturalHeight, needsResize: false };
  }
  const scale = maxDim / longest;
  return {
    width: Math.max(1, Math.round(naturalWidth * scale)),
    height: Math.max(1, Math.round(naturalHeight * scale)),
    needsResize: true,
  };
}

function putLruEntry(key: string, maxDim: number, entry: CachedDevelopBitmap): void {
  const hiRes = isHiResMaxDim(maxDim);
  const lru = hiRes ? hiResBitmapLru : bitmapLru;
  const maxEntries = hiRes ? MAX_HI_RES_BITMAPS : MAX_CACHED_BITMAPS;

  const existing = lru.get(key);
  if (existing) {
    if (existing.bitmap !== entry.bitmap) {
      existing.bitmap.close();
    }
    lru.delete(key);
  }
  lru.set(key, entry);
  while (lru.size > maxEntries) {
    const oldestKey = lru.keys().next().value;
    if (oldestKey === undefined) break;
    const oldest = lru.get(oldestKey);
    oldest?.bitmap.close();
    lru.delete(oldestKey);
  }
}

/**
 * Synchronous O(1) lookup for an already-decoded Studio preview bitmap (refreshes LRU recency).
 */
export function peekDevelopBitmap(src: string, maxDim: number): CachedDevelopBitmap | null {
  if (!src) return null;
  const lru = isHiResMaxDim(maxDim) ? hiResBitmapLru : bitmapLru;
  const key = makeCacheKey(src, maxDim);
  const hit = lru.get(key);
  if (!hit) return null;
  // Refresh LRU recency
  lru.delete(key);
  lru.set(key, hit);
  return hit;
}

function loadHtmlImage(src: string): Promise<HTMLImageElement | null> {
  return new Promise((resolve) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.decoding = 'async';
    img.onload = () => {
      if (img.naturalWidth > 0 && img.naturalHeight > 0) {
        resolve(img);
      } else {
        resolve(null);
      }
    };
    img.onerror = () => resolve(null);
    img.src = src;
  });
}

/**
 * Decodes and downscales `src` into a cached `ImageBitmap` capped at `maxDim`, coalescing
 * concurrent requests for the same frame (e.g. Split and Y|Y compare viewports).
 */
export function loadDevelopBitmap(
  src: string,
  maxDim: number,
): Promise<CachedDevelopBitmap | null> {
  if (!src) return Promise.resolve(null);
  const syncHit = peekDevelopBitmap(src, maxDim);
  if (syncHit) {
    return Promise.resolve(syncHit);
  }

  const key = makeCacheKey(src, maxDim);
  const existingPromise = inFlightLoads.get(key);
  if (existingPromise) {
    return existingPromise;
  }

  const promise = (async (): Promise<CachedDevelopBitmap | null> => {
    if (typeof createImageBitmap !== 'function') {
      return null;
    }
    const img = await loadHtmlImage(src);
    if (!img) return null;

    const naturalWidth = img.naturalWidth;
    const naturalHeight = img.naturalHeight;
    const target = computeTargetBitmapSize(naturalWidth, naturalHeight, maxDim);

    try {
      const bitmap = target.needsResize
        ? await createImageBitmap(img, {
            resizeWidth: target.width,
            resizeHeight: target.height,
            resizeQuality: 'high',
          })
        : await createImageBitmap(img);

      const entry: CachedDevelopBitmap = {
        bitmap,
        width: bitmap.width,
        height: bitmap.height,
        naturalWidth,
        naturalHeight,
      };
      putLruEntry(key, maxDim, entry);
      return entry;
    } catch {
      return null;
    }
  })().finally(() => {
    inFlightLoads.delete(key);
  });

  inFlightLoads.set(key, promise);
  return promise;
}

/**
 * Quietly pre-decodes a neighboring JPEG/PNG frame into the `ImageBitmap` LRU cache so
 * stepping forward or back in Studio uploads synchronously with zero decode stall.
 */
export function preloadDevelopBitmap(src: string, maxDim: number): void {
  if (!src) return;
  if (peekDevelopBitmap(src, maxDim)) return;
  void loadDevelopBitmap(src, maxDim);
}

/**
 * Releases all cached `ImageBitmap` GPU/CPU resources when switching projects.
 */
export function clearDevelopBitmapCache(): void {
  for (const entry of bitmapLru.values()) {
    entry.bitmap.close();
  }
  bitmapLru.clear();
  for (const entry of hiResBitmapLru.values()) {
    entry.bitmap.close();
  }
  hiResBitmapLru.clear();
  inFlightLoads.clear();
}
