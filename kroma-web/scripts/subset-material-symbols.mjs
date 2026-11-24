import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { collectUsedMaterialIcons } from './material-symbols-scanner.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const webRoot = path.resolve(__dirname, '..');
const srcRoot = path.join(webRoot, 'src');
const assetsDir = path.join(srcRoot, 'assets');
const outFontPath = path.join(assetsDir, 'material-symbols-outlined.woff2');
const outManifestPath = path.join(assetsDir, 'material-symbols-icons.json');

// Chrome UA guarantees Google Fonts CSS2 returns WOFF2 with OpenType GSUB ligatures intact.
const CHROME_UA =
  'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36';

async function main() {
  const icons = collectUsedMaterialIcons(srcRoot);
  if (icons.length === 0) {
    throw new Error(`No Material Symbols icons found under ${srcRoot}`);
  }

  // Google Fonts CSS2 requires `icon_names` to be strictly alphabetically sorted.
  const iconNamesParam = icons.join(',');
  const cssUrl =
    `https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20,400,0,0&icon_names=${encodeURIComponent(iconNamesParam)}`;

  console.log(`Requesting subsetted Material Symbols Outlined WOFF2 for ${icons.length} icons...`);
  const cssRes = await fetch(cssUrl, { headers: { 'User-Agent': CHROME_UA } });
  if (!cssRes.ok) {
    const body = await cssRes.text();
    throw new Error(`Google Fonts CSS2 request failed (${cssRes.status}): ${body.slice(0, 400)}`);
  }
  const cssText = await cssRes.text();
  const urlMatch = cssText.match(/url\((https:\/\/[^)]+)\)\s*format\(['"]?woff2['"]?\)/);
  if (!urlMatch) {
    throw new Error(`Could not locate WOFF2 URL in Google Fonts CSS response:\n${cssText}`);
  }

  const fontUrl = urlMatch[1];
  const fontRes = await fetch(fontUrl, { headers: { 'User-Agent': CHROME_UA } });
  if (!fontRes.ok) {
    throw new Error(`Failed downloading subsetted WOFF2 (${fontRes.status}) from ${fontUrl}`);
  }
  const bytes = Buffer.from(await fontRes.arrayBuffer());

  // Verify WOFF2 magic header ("wOF2")
  if (bytes.length < 48 || bytes.subarray(0, 4).toString('ascii') !== 'wOF2') {
    throw new Error(`Downloaded asset is not a valid WOFF2 font (${bytes.length} bytes)`);
  }

  fs.mkdirSync(assetsDir, { recursive: true });
  fs.writeFileSync(outFontPath, bytes);
  fs.writeFileSync(
    outManifestPath,
    `${JSON.stringify(
      {
        family: 'Material Symbols Outlined',
        axes: { opsz: 20, wght: 400, FILL: 0, GRAD: 0 },
        count: icons.length,
        bytes: bytes.length,
        icons,
      },
      null,
      2,
    )}\n`,
    'utf8',
  );

  console.log(
    `Wrote ${path.relative(webRoot, outFontPath)} (${(bytes.length / 1024).toFixed(1)} KiB for ${icons.length} icons)`,
  );
  console.log(`Wrote ${path.relative(webRoot, outManifestPath)}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
