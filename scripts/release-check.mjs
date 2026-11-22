#!/usr/bin/env node
/**
 * Kroma release integrity & version-sync validator.
 *
 * Usage:
 *   node scripts/release-check.mjs --check-versions [--expected-tag v0.1.0]
 *   node scripts/release-check.mjs --check-changelog [--expected-tag v0.1.0]
 *   node scripts/release-check.mjs --check-manifest
 *   node scripts/release-check.mjs --check-hygiene
 *   node scripts/release-check.mjs --all [--expected-tag v0.1.0]
 *   node scripts/release-check.mjs --extract-changelog [0.1.0]
 *   node scripts/release-check.mjs --bump 0.2.0
 */
import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(SCRIPT_DIR, '..');

const SEMVER_RE = /^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$/;
const SHA256_RE = /^[a-f0-9]{64}$/i;

const VERSION_TARGETS = [
  {
    label: 'Cargo.toml ([workspace.package].version)',
    relPath: 'Cargo.toml',
    read(text) {
      const section = text.match(/\[workspace\.package\][\s\S]*?(?=\n\[|$)/);
      const match = section?.[0].match(/^\s*version\s*=\s*"([^"]+)"/m);
      return match?.[1] ?? null;
    },
    write(text, nextVersion) {
      return text.replace(
        /(\[workspace\.package\][\s\S]*?^\s*version\s*=\s*")([^"]+)(")/m,
        `$1${nextVersion}$3`,
      );
    },
  },
  {
    label: 'kroma-tau/tauri.conf.json (version)',
    relPath: 'kroma-tau/tauri.conf.json',
    read(text) {
      return JSON.parse(text).version ?? null;
    },
    write(text, nextVersion) {
      return text.replace(/("version"\s*:\s*")([^"]+)(")/, `$1${nextVersion}$3`);
    },
  },
  {
    label: 'kroma-web/package.json (version)',
    relPath: 'kroma-web/package.json',
    read(text) {
      return JSON.parse(text).version ?? null;
    },
    write(text, nextVersion) {
      return text.replace(/("version"\s*:\s*")([^"]+)(")/, `$1${nextVersion}$3`);
    },
  },
  {
    label: 'kroma-web/src/domain/appUpdates.ts (KROMA_FALLBACK_VERSION)',
    relPath: 'kroma-web/src/domain/appUpdates.ts',
    read(text) {
      const match = text.match(/export const KROMA_FALLBACK_VERSION\s*=\s*'([^']+)'/);
      return match?.[1] ?? null;
    },
    write(text, nextVersion) {
      return text.replace(
        /(export const KROMA_FALLBACK_VERSION\s*=\s*')([^']+)(')/,
        `$1${nextVersion}$3`,
      );
    },
  },
];

function stripVPrefix(raw) {
  return String(raw ?? '').trim().replace(/^v/i, '');
}

function readRepoFile(relPath) {
  const abs = resolve(REPO_ROOT, relPath);
  if (!existsSync(abs)) {
    throw new Error(`Missing required file: ${relPath}`);
  }
  return readFileSync(abs, 'utf8');
}

function checkVersions(expectedTag) {
  const errors = [];
  const discovered = [];

  for (const target of VERSION_TARGETS) {
    const content = readRepoFile(target.relPath);
    const version = target.read(content);
    if (!version || !SEMVER_RE.test(version)) {
      errors.push(`${target.label}: invalid or missing semver ("${version ?? 'null'}")`);
    } else {
      discovered.push({ label: target.label, version });
    }
  }

  const uniqueVersions = Array.from(new Set(discovered.map((d) => d.version)));
  if (uniqueVersions.length > 1) {
    errors.push(
      `Version mismatch across files:\n` +
        discovered.map((d) => `  - ${d.label}: ${d.version}`).join('\n'),
    );
  }

  const canonical = uniqueVersions[0] ?? null;
  if (expectedTag) {
    const expected = stripVPrefix(expectedTag);
    if (!SEMVER_RE.test(expected)) {
      errors.push(`Expected tag "${expectedTag}" is not a valid semver tag (expected vX.Y.Z).`);
    } else if (canonical && canonical !== expected) {
      errors.push(
        `Tag version "${expected}" does not match workspace version "${canonical}". Bump versions before tagging.`,
      );
    }
  }

  return { ok: errors.length === 0, version: canonical, errors };
}

function extractChangelogSection(changelogText, targetVersion) {
  const normalized = changelogText.replace(/\r\n?/g, '\n');
  const cleanTarget = stripVPrefix(targetVersion);
  const escaped = cleanTarget.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const headingRe = new RegExp(`^##\\s+\\[?v?${escaped}\\]?(?:\\b|$)`, 'im');
  const lines = normalized.split('\n');

  for (let i = 0; i < lines.length; i++) {
    if (headingRe.test(lines[i])) {
      let end = lines.length;
      for (let j = i + 1; j < lines.length; j++) {
        if (/^##\s+/.test(lines[j])) {
          end = j;
          break;
        }
      }
      return lines.slice(i, end).join('\n').trim();
    }
  }
  return null;
}

function checkChangelog(targetVersion) {
  const errors = [];
  const abs = resolve(REPO_ROOT, 'CHANGELOG.md');
  if (!existsSync(abs)) {
    return { ok: false, errors: ['Missing root CHANGELOG.md file.'] };
  }
  const text = readFileSync(abs, 'utf8');
  const section = extractChangelogSection(text, targetVersion);
  if (!section) {
    errors.push(`CHANGELOG.md is missing a "## [${targetVersion}]" section.`);
  } else {
    const bodyLines = section
      .split('\n')
      .slice(1)
      .map((l) => l.trim())
      .filter((l) => l.length > 0);
    if (bodyLines.length === 0) {
      errors.push(`CHANGELOG.md section for v${targetVersion} has no release notes.`);
    }
  }
  return { ok: errors.length === 0, errors };
}

function checkManifest() {
  const errors = [];
  let manifest;
  try {
    manifest = JSON.parse(readRepoFile('models/models.json'));
  } catch (err) {
    return { ok: false, errors: [`Could not parse models/models.json: ${String(err)}`] };
  }

  if (manifest.schemaVersion !== 1) {
    errors.push(`models/models.json: expected schemaVersion 1, got ${manifest.schemaVersion}`);
  }
  if (typeof manifest.releaseTag !== 'string' || !manifest.releaseTag.trim()) {
    errors.push('models/models.json: missing releaseTag');
  }
  if (!Array.isArray(manifest.models) || manifest.models.length === 0) {
    errors.push('models/models.json: models array must be non-empty');
  } else {
    const seenIds = new Set();
    for (const model of manifest.models) {
      const id = String(model?.id ?? '');
      if (!id) {
        errors.push('models/models.json: model entry missing id');
        continue;
      }
      if (seenIds.has(id)) {
        errors.push(`models/models.json: duplicate model id "${id}"`);
      }
      seenIds.add(id);

      if (typeof model.name !== 'string' || !model.name.trim()) {
        errors.push(`models/models.json (${id}): missing name`);
      }
      if (typeof model.version !== 'string' || !model.version.trim()) {
        errors.push(`models/models.json (${id}): missing version`);
      }
      if (
        typeof model.relativePath !== 'string' ||
        !model.relativePath.trim() ||
        model.relativePath.includes('..') ||
        model.relativePath.startsWith('/') ||
        model.relativePath.startsWith('\\')
      ) {
        errors.push(`models/models.json (${id}): invalid relativePath "${model.relativePath}"`);
      }
      if (typeof model.url !== 'string' || !model.url.startsWith('https://')) {
        errors.push(`models/models.json (${id}): url must use https://`);
      }
      if (typeof model.sizeBytes !== 'number' || !Number.isInteger(model.sizeBytes) || model.sizeBytes <= 0) {
        errors.push(`models/models.json (${id}): sizeBytes must be a positive integer`);
      }
      if (typeof model.sha256 !== 'string' || !SHA256_RE.test(model.sha256)) {
        errors.push(`models/models.json (${id}): sha256 must be a 64-character hex digest`);
      }
    }
  }

  return { ok: errors.length === 0, count: manifest.models?.length ?? 0, errors };
}

function checkGitHygiene() {
  const errors = [];
  if (!existsSync(resolve(REPO_ROOT, '.git'))) {
    return { ok: true, skipped: true, errors };
  }
  try {
    const tracked = execFileSync('git', ['ls-files'], {
      cwd: REPO_ROOT,
      encoding: 'utf8',
    })
      .split(/\r?\n/)
      .filter(Boolean);
    const badFiles = tracked.filter((file) => /\.(onnx|bin|pt|pth|safetensors)$/i.test(file));
    if (badFiles.length > 0) {
      errors.push(
        `Large AI weight files must not be committed to Git (host on models-v1 release instead):\n` +
          badFiles.map((f) => `  - ${f}`).join('\n'),
      );
    }
  } catch (err) {
    errors.push(`Git hygiene check failed: ${String(err)}`);
  }
  return { ok: errors.length === 0, skipped: false, errors };
}

function bumpAllVersions(nextRaw) {
  const nextVersion = stripVPrefix(nextRaw);
  if (!SEMVER_RE.test(nextVersion)) {
    console.error(`Error: "${nextRaw}" is not a valid semver string (expected e.g. 0.2.0).`);
    process.exit(1);
  }
  for (const target of VERSION_TARGETS) {
    const abs = resolve(REPO_ROOT, target.relPath);
    const currentText = readRepoFile(target.relPath);
    const updatedText = target.write(currentText, nextVersion);
    if (updatedText === currentText && target.read(currentText) !== nextVersion) {
      console.error(`Error: could not update version in ${target.relPath}`);
      process.exit(1);
    }
    writeFileSync(abs, updatedText, 'utf8');
    console.log(`Updated ${target.relPath} -> ${nextVersion}`);
  }
  console.log('\nNext: update Cargo.lock (`cargo metadata --format-version 1 > NUL`) and add a section in CHANGELOG.md.');
}

function main() {
  const args = process.argv.slice(2);
  const hasFlag = (flag) => args.includes(flag);
  const getArg = (flag) => {
    const idx = args.indexOf(flag);
    return idx >= 0 && idx + 1 < args.length ? args[idx + 1] : null;
  };

  if (hasFlag('--bump')) {
    const next = getArg('--bump');
    if (!next) {
      console.error('Usage: node scripts/release-check.mjs --bump <X.Y.Z>');
      process.exit(1);
    }
    bumpAllVersions(next);
    return;
  }

  if (hasFlag('--extract-changelog')) {
    const requested = getArg('--extract-changelog');
    const verCheck = checkVersions(null);
    const targetVersion = requested ? stripVPrefix(requested) : verCheck.version;
    if (!targetVersion) {
      console.error('Could not determine target version for changelog extraction.');
      process.exit(1);
    }
    const text = readRepoFile('CHANGELOG.md');
    const section = extractChangelogSection(text, targetVersion);
    if (!section) {
      console.error(`No CHANGELOG.md section found for version ${targetVersion}`);
      process.exit(1);
    }
    process.stdout.write(`${section}\n`);
    return;
  }

  const runAll = hasFlag('--all') || args.length === 0;
  const doVersions = runAll || hasFlag('--check-versions');
  const doChangelog = runAll || hasFlag('--check-changelog');
  const doManifest = runAll || hasFlag('--check-manifest');
  const doHygiene = runAll || hasFlag('--check-hygiene');
  const expectedTag = getArg('--expected-tag');

  const allErrors = [];
  let resolvedVersion = expectedTag ? stripVPrefix(expectedTag) : null;

  if (doVersions) {
    const res = checkVersions(expectedTag);
    if (res.version) resolvedVersion = res.version;
    if (res.ok) {
      console.log(`✓ Versions synced at v${res.version} across ${VERSION_TARGETS.length} files`);
    } else {
      allErrors.push(...res.errors);
    }
  }

  if (doChangelog) {
    if (!resolvedVersion) {
      resolvedVersion = checkVersions(null).version;
    }
    const res = checkChangelog(resolvedVersion ?? '');
    if (res.ok) {
      console.log(`✓ CHANGELOG.md contains release notes for v${resolvedVersion}`);
    } else {
      allErrors.push(...res.errors);
    }
  }

  if (doManifest) {
    const res = checkManifest();
    if (res.ok) {
      console.log(`✓ models/models.json valid (${res.count} AI model entries)`);
    } else {
      allErrors.push(...res.errors);
    }
  }

  if (doHygiene) {
    const res = checkGitHygiene();
    if (res.ok) {
      console.log(
        res.skipped
          ? '✓ Git hygiene check skipped (no .git directory yet)'
          : '✓ Git hygiene clean (no tracked .onnx/weight binaries)',
      );
    } else {
      allErrors.push(...res.errors);
    }
  }

  if (allErrors.length > 0) {
    console.error('\nRelease check failed:');
    for (const err of allErrors) {
      console.error(`✗ ${err}`);
    }
    process.exit(1);
  }
}

main();
