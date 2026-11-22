<div align="center">
  <img src="./assets/kroma-logo.png" alt="Kroma logo" width="96" />
  <h1>Kroma</h1>
</div>

<p align="center">
  <img src="./assets/sample.png" alt="Kroma 32-Bit RAW Studio preview" width="100%" />
  <br />
  <sub>Sample RAW portrait (<code>IMG_7426.CR2</code>) courtesy of <a href="https://www.signatureedits.com/free-raw-photos/">Signature Edits</a> (<a href="https://www.instagram.com/signatureeditsco/">@signatureeditsco</a>).</sub>
</p>

## About

Kroma is a **local-first desktop RAW photo editor and portrait workflow studio** with a **32-bit GPU develop engine**, **AI face organization**, and **offline whiteboard OCR** built for fast high-volume shoots.

> [!NOTE]
> All AI models (face detection, face recognition, Slate OCR, and subject masking) run 100% locally on your device. On first launch, Kroma automatically downloads and SHA-256 verifies the model weights from [GitHub Releases](https://github.com/ByteKen/kroma/releases).

> [!IMPORTANT]
> Kroma is strictly **non-destructive**. Original RAW and image files are never modified; all develop edits, masks, and subject groupings are stored in lightweight `.kroma.json` sidecar metadata.

## Why Kroma?

Volume portrait photography generates thousands of RAW images across hundreds of subjects in a single session. In workflows like school picture days, graduations, and headshot events, photographers photograph a whiteboard slate with the person's name before taking their portraits.

Sorting through those files afterwards requires matching names from the slates to the photos, separating each person into folders, and culling duplicates before grading and delivery.

Kroma brings this entire process together on the local workstation:

1. **Automatic Slate and Subject Organization:** Built in OCR reads names directly from whiteboard slates and pairs them with local face clustering to group each person's portraits automatically.
2. **Private and On Device:** Face recognition, background removal, and slate OCR run entirely on the local GPU. No client photos or data leave the machine.
3. **32 Bit Linear Grading:** Develop adjustments process in 32 bit linear floating point on the GPU to preserve highlight detail and smooth color transitions during color work.
4. **Portable Sidecar Storage:** All edits, masks, and subject groupings save into `.kroma.json` sidecar files stored directly with the original images, keeping projects simple to backup or move.

## Features

- **Cross-Platform Native App:** Built with Rust and Tauri v2 for Windows and macOS, with recent projects catalog, pinning, and automatic release update checks
- **32-Bit Wide-Gamut Develop Engine:** Unified WGSL compute pipeline running on WebGPU (live studio, with automatic WebGL2 fallback) and `wgpu` DX12/Metal (batch export) in Linear Rec.2020 color space
- **Full RAW & Image Support:** Fast native decoding for 29+ camera RAW formats (ARW, CR2, CR3, NEF, RAF, DNG, ORF, RW2, and more) alongside JPEG, PNG, and TIFF
- **Non-Destructive Grading & Presets:** White balance, exposure, tone curves, HSL color mixer, 3-way color grading wheels, film grain, crop/level, built-in & custom presets, and 3D `.cube` LUT support
- **AI Masking & Backdrop Replacement:** Local `RMBG-1.4` subject/background segmentation, non-destructive studio background replacement (solid, linear/radial gradient, or transparent), plus brush, linear/radial gradient, and luminance/color range masks
- **Batch Sync & Portrait Calibration:** Selective module sync, copy/paste recipes, relative exposure/WB nudges across multi-selected frames, 1-click eye-horizon auto-leveling, and 1-click white balance calibration from whiteboard/slate frames
- **Live Scopes & Comparison Views:** Real-time RGB histogram, highlight/shadow clipping overlays, Split Before/After, Side-by-Side (`Y|Y`), Focus Mode, and full-screen Lightbox
- **Face Organizer & Manual Culling:** On-demand subject clustering (YuNet + ArcFace R50 512-D) with Time Assist, Ambiguous Review Queue, Manual Match Split View, drag-and-drop merge/split/move, smart bins, nested virtual folders, and independent Develop & Organization undo/redo
- **Slate OCR:** Local PP-OCRv5 whiteboard and name-card scanning with high-resolution RAW fallback to automatically extract subject names and codes
- **Selective Batch Export:** Export full shoots, specific folders, or selected frames as graded JPEG, PNG, or 16-bit TIFF with optional original RAW pairing, custom naming patterns, automatic slate exclusion/archiving, and real-time output size estimation
- **Offline-First & Configurable AI:** Zero cloud uploads and zero subscriptions; hardware-accelerated via DirectML (Windows) and CoreML (macOS), with custom `.onnx` model inspection/swapping and configurable cache directories

## Showcase

### 32-Bit RAW Develop (Before vs. After)

<table>
  <thead>
    <tr>
      <th width="50%">Original Camera RAW (<code>IMG_7426.CR2</code>)</th>
      <th width="50%">32-Bit Linear Rec.2020 Grade in Kroma</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><img src="./assets/showcase/raw-develop-before.jpg" alt="Original unedited camera RAW portrait" width="100%" /></td>
      <td><img src="./assets/showcase/raw-develop-after.jpg" alt="32-bit graded camera RAW portrait exported from Kroma" width="100%" /></td>
    </tr>
  </tbody>
</table>

### AI Masking & Background Removal

<p align="center">
  <img src="./assets/showcase/ai-masking-background-removal.png" alt="Kroma AI Masking and Background Removal in 32-Bit RAW Studio" width="100%" />
</p>

### On-Demand Face Organizer & Subject Clustering

<table>
  <thead>
    <tr>
      <th width="50%">1. All Imported Photos (Unclustered)</th>
      <th width="50%">2. Clustered by Face (YuNet + ArcFace 512-D)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><img src="./assets/showcase/face-organizer-import.png" alt="Kroma unclustered import photo grid" width="100%" /></td>
      <td><img src="./assets/showcase/face-organizer-clustered.png" alt="Kroma face organizer clustered subjects with landmarks" width="100%" /></td>
    </tr>
  </tbody>
</table>

### Side-by-Side Identity Review & Slate OCR

<table>
  <thead>
    <tr>
      <th width="50%">Ambiguous Identity Review Queue</th>
      <th width="50%">Local PP-OCRv5 Whiteboard Slate Scanner</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><img src="./assets/showcase/identity-review-queue.png" alt="Kroma side-by-side identity review queue" width="100%" /></td>
      <td><img src="./assets/showcase/slate-ocr-whiteboard.png" alt="Kroma Slate OCR extracting subject name from whiteboard" width="100%" /></td>
    </tr>
  </tbody>
</table>

## Installation

Download the latest installer or portable archive for your OS from the [Releases](https://github.com/ByteKen/kroma/releases) page.

## Development

Kroma is built with **Rust**, **Tauri v2**, **React 19**, **TypeScript**, and **WGSL**.

### Prerequisites

- [Rust](https://www.rust-lang.org/tools/install) (Edition 2024 / stable toolchain)
- [Node.js](https://nodejs.org/) (v20+)
- Platform Tauri v2 build dependencies ([Tauri prerequisites](https://v2.tauri.app/start/prerequisites/))

### Run Locally

```bash
# Install frontend dependencies
npm --prefix kroma-web install

# Launch the desktop app in development mode
npm --prefix kroma-web run tauri:dev

# Build a production desktop bundle
npm --prefix kroma-web run tauri:build
```

> [!NOTE]
> See [CHANGELOG.md](./CHANGELOG.md) for release history and [models/models.json](./models/models.json) for the bundled AI model manifest.

## License

Kroma's source code is licensed under the [GNU General Public License v3.0](./LICENSE).

> [!NOTE]
> The default ONNX AI models downloaded on first launch remain governed by their respective upstream licenses: **YuNet** (MIT), **PP-OCRv5** (Apache-2.0), **InsightFace ArcFace `w600k_r50`** (non-commercial research), and **BRIA `RMBG-1.4`** (non-commercial). You can configure custom `.onnx` models in **Settings → Models**.
