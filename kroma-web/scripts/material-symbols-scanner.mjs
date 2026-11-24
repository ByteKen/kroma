import fs from 'node:fs';
import path from 'node:path';

/**
 * Scans `srcRoot` for every Material Symbols Outlined ligature name rendered in TSX/TS
 * (literal `<span className="...material-symbols-outlined...">name</span>`, ternary expressions
 * inside those spans, `icon="..."`, `icon: '...'`, and ternary `icon={...}` props).
 *
 * Returns a sorted, deduplicated array of icon names.
 */
export function collectUsedMaterialIcons(srcRoot) {
  const files = [];
  const walk = (dir) => {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        walk(fullPath);
      } else if (/\.(ts|tsx)$/.test(entry.name)) {
        files.push(fullPath);
      }
    }
  };
  walk(srcRoot);

  const icons = new Set();

  for (const filePath of files) {
    const src = fs.readFileSync(filePath, 'utf8');

    // 1. Direct <span ...material-symbols-outlined...>...</span>
    for (const match of src.matchAll(
      /<span[^>]*material-symbols-outlined[^>]*>([\s\S]*?)<\/span>/g,
    )) {
      const inner = match[1].trim();
      if (/^[a-z0-9_]+$/.test(inner)) {
        icons.add(inner);
      } else if (/^\{[\s\S]*\}$/.test(inner)) {
        const expr = inner.slice(1, -1).trim();
        for (const q of expr.matchAll(/[?:]\s*['"]([a-z0-9_]+)['"]/g)) {
          icons.add(q[1]);
        }
      }
    }

    // 2. Literal `icon="..."` or `icon: '...'`
    for (const match of src.matchAll(/\bicon\s*[:=]\s*['"]([a-z0-9_]+)['"]/g)) {
      icons.add(match[1]);
    }

    // 3. Ternary `icon={cond ? 'a' : 'b'}`
    for (const match of src.matchAll(/\bicon\s*=\s*\{([^}]+)\}/g)) {
      for (const q of match[1].matchAll(/[?:]\s*['"]([a-z0-9_]+)['"]/g)) {
        icons.add(q[1]);
      }
    }
  }

  return [...icons].sort();
}
