import { isFaceOnlySlateCode } from './identity';

export interface OutputNamingRule {
  readonly pattern: string;
  readonly excludeSlateFromClientPack: boolean;
}

export type ExportFileNamingMode = 'original' | 'sequential';

export const DEFAULT_OUTPUT_NAMING_RULE: OutputNamingRule = {
  pattern: '/[Slate_Name]/[Original_Name].jpg',
  excludeSlateFromClientPack: true,
};

export function buildSimpleExportPattern(
  mode: ExportFileNamingMode,
  subfolderPerSubject: boolean,
): string {
  if (mode === 'original') {
    return subfolderPerSubject ? '/[Slate_Name]/[Original_Name].jpg' : '/[Original_Name].jpg';
  }
  return subfolderPerSubject ? '/[Slate_Name]/IMG_[Seq].jpg' : '/[Slate_Name]_[Seq].jpg';
}

/**
 * Returns per-cluster export subject names, appending each colliding cluster's generated code
 * (e.g. `Santos, Maria_001`, `Santos, Maria_002`) ONLY when two subjects share the same sanitized
 * name and the pattern does not already contain `[ID]`.
 */
export function resolveUniqueExportSubjectNames(
  clusters: readonly {
    readonly anchor: {
      readonly id: string;
      readonly subjectName: string;
      readonly subjectCode?: string;
      readonly slateCode: string;
    };
  }[],
  pattern: string,
): ReadonlyMap<string, string> {
  const result = new Map<string, string>();
  const patternIncludesId = pattern.includes('[ID]');
  const nameCounts = new Map<string, number>();

  for (const cluster of clusters) {
    const key = sanitizePathSegment(cluster.anchor.subjectName).toLowerCase();
    nameCounts.set(key, (nameCounts.get(key) ?? 0) + 1);
  }

  clusters.forEach((cluster, idx) => {
    const key = sanitizePathSegment(cluster.anchor.subjectName).toLowerCase();
    const collides = !patternIncludesId && (nameCounts.get(key) ?? 0) > 1;
    if (!collides) {
      result.set(cluster.anchor.id, cluster.anchor.subjectName);
      return;
    }
    const rawCode =
      cluster.anchor.subjectCode ??
      (!isFaceOnlySlateCode(cluster.anchor.slateCode) ? cluster.anchor.slateCode : '');
    const suffix = sanitizePathSegment(rawCode) || String(idx + 1).padStart(3, '0');
    result.set(cluster.anchor.id, `${cluster.anchor.subjectName}_${suffix}`);
  });

  return result;
}

export function inferSimpleExportNaming(pattern: string): {
  readonly mode: ExportFileNamingMode;
  readonly subfolderPerSubject: boolean;
} {
  const mode: ExportFileNamingMode =
    pattern.includes('[Original_Name]') || pattern.includes('[Original]')
      ? 'original'
      : 'sequential';
  const normalized = pattern.replace(/^\/+/, '');
  const slashIdx = normalized.lastIndexOf('/');
  const dirPart = slashIdx >= 0 ? normalized.slice(0, slashIdx) : '';
  const subfolderPerSubject =
    dirPart.includes('[Slate_Name]') || dirPart.includes('[Subject_Name]');
  return { mode, subfolderPerSubject };
}

function sanitizePathSegment(raw: string): string {
  return raw
    .trim()
    .replace(/[^a-zA-Z0-9_-]+/g, '_')
    .replace(/^_+|_+$/g, '');
}

function extractFileStem(fileName?: string): string {
  if (!fileName) return '';
  const base = fileName.split(/[\\/]/).pop() ?? fileName;
  const dot = base.lastIndexOf('.');
  return dot > 0 ? base.slice(0, dot) : base;
}

export function formatExportPreviewPath(
  rule: OutputNamingRule,
  subjectName: string,
  slateCode: string,
  sequenceNumber: number,
  virtualFolderSegments: readonly string[] = [],
  originalFileName?: string,
): string {
  const safeName = sanitizePathSegment(subjectName) || 'Unknown_Subject';
  const hasSlateCode = !isFaceOnlySlateCode(slateCode);
  const safeCode = hasSlateCode ? sanitizePathSegment(slateCode) : '';
  const paddedSeq = String(Math.max(1, Math.floor(sequenceNumber))).padStart(3, '0');
  const safeOriginal =
    sanitizePathSegment(extractFileStem(originalFileName)) || `IMG_${paddedSeq}`;
  const safeFolderPath = virtualFolderSegments
    .map((seg) => sanitizePathSegment(seg))
    .filter((s) => s.length > 0)
    .join('/');
  const hasFolderToken =
    rule.pattern.includes('[Folder]') || rule.pattern.includes('[Department]');

  let rendered = rule.pattern
    .replaceAll('[Seq]', paddedSeq)
    .replaceAll('001', paddedSeq)
    .replaceAll('[Original_Name]', safeOriginal)
    .replaceAll('[Original]', safeOriginal)
    .replaceAll('[Slate_Name]', safeName)
    .replaceAll('[Subject_Name]', safeName);

  if (safeCode.length > 0) {
    rendered = rendered.replaceAll('[ID]', safeCode);
  } else {
    rendered = rendered
      .replaceAll('_[ID]', '')
      .replaceAll('[ID]_', '')
      .replaceAll('-[ID]', '')
      .replaceAll('[ID]-', '')
      .replaceAll('[ID]', '');
  }

  if (hasFolderToken) {
    if (safeFolderPath.length > 0) {
      rendered = rendered
        .replaceAll('[Folder]', safeFolderPath)
        .replaceAll('[Department]', safeFolderPath);
    } else {
      rendered = rendered
        .replaceAll('/[Folder]/', '/')
        .replaceAll('[Folder]/', '')
        .replaceAll('/[Folder]', '')
        .replaceAll('[Folder]', '')
        .replaceAll('/[Department]/', '/')
        .replaceAll('[Department]/', '')
        .replaceAll('/[Department]', '')
        .replaceAll('[Department]', '');
    }
  } else if (safeFolderPath.length > 0) {
    const leadingSlash = rendered.startsWith('/') ? '/' : '';
    rendered = `${leadingSlash}${safeFolderPath}/${rendered.replace(/^\/+/, '')}`;
  }

  return rendered.replace(/\/{2,}/g, '/');
}
