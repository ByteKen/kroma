# Changelog

All notable changes to **Kroma** are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-11-25

### Added
- **Non-Destructive Retouch Suite (`Retouch` Tab in Studio):**
  - **Portrait Retouching (Frequency Separation):** GPU-accelerated skin smoothing with texture preservation, skin tone evenness, shine removal, under-eye brightening, eye enhancement, teeth whitening, and lip enhancement—powered by an optional **BiSeNet Face Parsing** model (`parsing_resnet18.onnx`) with automatic built-in YuNet 5-point fallback, running across WebGPU, WebGL2, and `wgpu` batch export.
  - **Spot Heal & Clone Brush:** Non-destructive blemish and stray-hair healing with interactive pin overlays (`H`), draggable sample sources, adjustable radius/feather/opacity, and `[` / `]` brush resizing.
  - **AI Object Removal (`Remove` Brush):** Local generative object removal powered by an optional **LaMa Inpainting** model (`lama_fp32.onnx`) operating directly on Studio's 32-bit graded RAW working buffer, with mirror-padded square context crops, sub-pixel Catmull-Rom patch alignment, perimeter tone adaptation, and variance-matched micro-grain.
  - **On-Demand Optional Retouch Models (`Settings → Models`):** Download, verify (SHA-256), or swap custom `.onnx` weights for BiSeNet Face Parsing and LaMa Inpainting directly inside Settings without bloating initial startup downloads.
  - **Batch Sync Integration:** Selectively copy Portrait Retouching (auto-adapting to each subject's face) and Spot Heal patches across selected frames via `Ctrl+Shift+S`.
- **Studio Folders Search & Expandable Photo Dropdown:** Added a `Ctrl+K` search bar to the Studio `Folders` sidebar that filters by both folder path and contained photo filenames (auto-expanding matching folders), plus per-folder dropdown chevrons showing alphabetically ordered photo names with an ultra-light debounced hover preview.
- **Alphabetical Sorting Across All Lists:** Made natural case-insensitive A–Z sorting the default across Studio `Folders`, `Subjects`, and filmstrip, Face Organizer (`A–Z` added as the default sort option in the sidebar and Settings), Virtual Folders, Slate OCR, Batch Export, Review Queue, and Merge Subject dialogs.

### Changed
- **Latest-Wins RAW Decode Scheduler & Priority Preemption:** Bounded heavy RAW Bayer demosaics to a single latest-wins slot with active-over-prefetch priority and a 45 ms uncached settle debounce so rapidly clicking across subjects or scrubbing with arrow keys never launches competing demosaics or drops UI FPS—while preserving instant RAM LRU (~0.1 ms) and SSD `.kr16` (~8–10 ms) cache hits.
- **Minimalist Studio Workspace Redesign:** Enlarged the bottom filmstrip (`96×116 px` cards with hover-reveal sync checkboxes and wider scroll track), refined Develop sliders and hover-to-edit numeric scrubbers, decluttered top/bottom viewport controls and inspector tabs, and removed the redundant `Open Photo(s)` toolbar button from the Home screen.

### Fixed
- **AI Object Removal Color & Brush Accuracy:** Fixed RAW vs. 8-bit preview exposure/color drift when generating LaMa inpaint patches, tightened mask feathering, and aligned the Remove brush cursor to match the exact painted mask area.
- **32-Bit RAW Loading & Thumbnail Progress Reliability:** Fixed a Vite HMR gateway reset that could interrupt 32-bit RAW decoding during dev sessions, guarded 4K 1:1 loupe upgrades from preempting Fit-view decodes on the same photo, and eliminated stale `0/1` thumbnail rendering progress pills across overlapping edits and cancellations.

## [0.2.0] - 2026-11-24

### Added
- **Single Photo Mode & Direct File Opening (`Ctrl+Shift+O`):** Open an individual photo or custom file selection directly from the Start Screen without scanning an entire shoot folder. When editing a single photo, Kroma automatically focuses the workspace on Studio Develop and Export—hiding Face Organizer, Slate OCR, filmstrip, and batch sync controls—with a one-click **Open Parent Folder** shortcut whenever you want to expand into the full shoot.
- **Full-Stage Darkroom Canvas & Free-Form GPU Zoom/Pan:** Replaced clipped frame-box zoom with an edge-to-edge full-stage GPU viewport across Single, Split Wipe, and Side-by-Side (`Y|Y`) modes. Zoom smoothly toward the cursor with `Ctrl+Scroll` / `Cmd+Scroll`, double-click or press `Z` to toggle `Fit ↔ 100% (1:1)`, pan freely in 2D across the stage, and automatically upgrade zoomed details to crisp 4K+ resolution on demand (`4096px` in Single loupe, `3072px` in dual compare) with bounded memory.
- **Unified Start Screen Action Card:** Combined **Open Folder** (`Ctrl+O`) and **Open Photo(s)** (`Ctrl+Shift+O`) into a single viewfinder action card on the Welcome screen.
- **Project Community Templates & Contributing Guide:** Added Kroma-specific GitHub bug report and feature request forms, PR template, and `CONTRIBUTING.md` guidelines.

### Changed
- **120 Hz Zero-Lag Develop Scrubbing & Fast Present Path:** Decoupled live slider, curve, color-wheel, crop, and brush drags into an external store (`liveDevelopDraftStore`) to eliminate full-shell React re-renders during adjustments, and split WebGPU rendering into a present-only fast path (~0.2 ms/frame) so panning, zooming, and clipping toggles skip heavy compute passes.
- **Faster JPEG/PNG Browsing & Tab Switching:** Added a bounded LRU `ImageBitmap` cache with neighbor pre-decoding (`developBitmapCache`) for instant JPEG switching, plus cached `1024px` web preview proxies (`>= 300 KB`) and virtualized review queues to eliminate FPS drops when switching workspaces.
- **Faster Batch Exports & Lower Memory Churn:** Added GPU texture, uniform, staging-buffer, and 3D LUT pooling plus parallel Rayon pack/unpack in the `wgpu` export engine, cached `.cube` LUT parsing, parallel Rayon area-averaging preview downscaling, cached 384 px RAM histogram proxies for JPEG/PNG photos, and release-profile optimizations (`lto = "thin"`, `codegen-units = 1`, symbol stripping).
- **Leaner Frontend Bundle (-3.9 MB):** Subsetted the bundled offline Material Symbols icon font from `4,026 kB` to `11.19 kB` (-99.7%) with an automated Vitest manifest check, and split React, Framer Motion, and Tauri runtime chunks in Vite.
- **Clean Sidecar Storage for Downloads & Desktop:** Kept single-photo `.kroma.json` sidecars inside Kroma's managed app-data directory when opening standalone files from shared folders like Downloads or Desktop so personal folders stay uncluttered.

### Fixed
- **Batch Export CPU & RAM Spikes:** Bounded export worker concurrency and recycled `wgpu` scratch resources to eliminate memory surges during large multi-photo exports.
- **Deep-Zoom WebGPU Stability:** Added 512-row chunked WebGPU texture uploads (`<= 16 MB` per transfer), automatic `device.lost` recovery, and async OOM/validation error scopes so zooming deeply on high-megapixel RAW files never blanks the viewport.
- **Virtualized Grid Scroll Jitter:** Quantized virtual grid/strip/list scroll ranges via `requestAnimationFrame` and memoized photo cluster cards for steady scrolling across large shoots.

## [0.1.2] - 2026-11-23

### Added
- **Custom 3D `.cube` LUT File Import:** Import arbitrary `.cube` 3D look-up tables directly from disk via native file dialog. Custom LUTs are parsed, stored in local storage, and bound dynamically to WebGPU compute storage buffers and WebGL2 3D textures.
- **Relocated 3D LUT Controls to Presets Tab:** Moved the 3D `.cube` LUT profile selector to the Presets (`P`) inspector tab alongside built-in film emulations (Kodak Portra 400, Fuji Pro 400H, Kroma Studio Neutral, Ilford HP5 Mono) and user-imported custom LUTs.

### Changed
- **Showcase & Workflow Documentation:** Added before/after 32-bit linear RAW grading comparisons, AI background masking, face clustering, identity review, and slate OCR previews with sample photo credits.

### Fixed
- **Thumbnail Rendering for Non-RAW Shoots:** Prevented unnecessary 32-bit CPU background rendering passes when viewing folders containing only 8-bit image formats (JPEG, PNG).
- **Release Body Cleanup:** Removed redundant operating system table from the automated GitHub release workflow.

## [0.1.1] - 2026-11-22

### Added
- **Optional 32-Bit Rendered Thumbnails (`Settings → Performance`):** Choose between `Fast (Embedded + Instant GPU)`, `Edited Only (32-Bit)` (default), and `All Thumbnails (32-Bit)` to display 32-bit Linear Rec.2020 rendered thumbnails across Face Organizer, Lightbox, Studio Filmstrip, Sync, and Export previews.
- **Instant Studio RAW Caching & Neighbor Pre-Decoding:** Added a synchronous 12-photo RAM LRU cache, a bounded 24-photo SSD working-buffer cache (`.kr16`), automatic background neighbor pre-decoding (`+1`, `-1`, `+2`), and zero-copy `rgba16float` WebGPU texture uploads into the 32-bit `rgba32float` compute pipeline.
- **Bottom-Bar 32-Bit Studio & Preview Telemetry:** Moved the 32-bit RAW decode/ready indicator off the photo canvas into the bottom-right status bar alongside a live `32-bit Previews X/Y` background rendering pill.

### Changed
- **Fused Parallel Rayon Rec.2020 RAW Conversion:** Combined box-filter downscaling and Rec.709→Rec.2020 linear matrix conversion into a single multi-core pass, cutting cold RAW decode memory and latency in half.
- **Slate Photos Included in Studio Sync:** Allowed slate reference photos to be checked in the Studio filmstrip and included in `Selected`, `Current Folder / Subject`, and `Entire Project` recipe syncs.

### Fixed
- **Portrait RAW Orientation in 32-Bit Studio & Export:** Added explicit EXIF IFD orientation fallback when camera RAW decoders report `Orientation::Normal`, fixing sideways portrait `.ARW`, `.CR2`, and `.NEF` frames after 32-bit decoding.
- **Subfolder `.kroma.json` Sidecar Reloading:** Fixed sanitized sidecar filename mapping so saved adjustments inside subfolders with spaces, commas, or symbols restore reliably when reopening a project.
- **Same-Stem RAW + JPEG Export Filename Collision:** Disambiguated duplicate output filenames within the same folder (e.g., `DSC05928.ARW` + `DSC05928.jpg` both exporting to `.jpg`) using `_2`, `_3` suffixes so no photo is overwritten.
- **Desktop External Browser Links (`Download Update` & `About`):** Added native in-app update installer downloading & launching (`download_and_install_app_update` with live progress bar) plus a native OS browser launcher (`open_external_url`) for external links on Windows and macOS.

## [0.1.0] - 2026-11-22

### Added
- **32-Bit Wide-Gamut RAW & WGSL GPU Engine:** Unified WGSL compute pipeline running in Linear Rec.2020 on WebGPU (live studio with automatic WebGL2 fallback) and `wgpu` (DX12 on Windows, Metal on macOS) for batch export—supporting 29+ RAW formats, tone curves, HSL color mixer, 3-way color grading, 3D `.cube` LUTs, film grain, crop/level, local AI/manual masking, and non-destructive studio backdrop replacement.
- **Cross-Platform Desktop (Windows & macOS):** Native Tauri v2 builds for Windows (DirectML) and macOS Apple Silicon (CoreML) with platform-native window controls and keyboard shortcuts.
- **Face Organizer & Virtual Folders:** On-demand face clustering (YuNet + ArcFace R50 512-D), Time Assist, Ambiguous Review Queue, Manual Match Split View, nested virtual folders, and independent Develop & Organization undo/redo history.
- **Slate OCR:** Local PP-OCRv5 whiteboard scanning with smart high-resolution RAW fallback for automatic subject naming.
- **Selective Batch Export:** Export single photos, specific folders, or full shoots as graded JPEG, PNG, or 16-bit TIFF with optional original RAW pairing, custom naming patterns, slate archiving, and real-time output size estimation.
- **Manifest-Driven AI Bootstrapping:** Automatic verification and streaming SHA-256-verified model downloads on first launch via `models/models.json` (`models-v1` release assets).
- **Interactive Settings & Update Gate:** 6-tab Settings modal (`General`, `Performance`, `Models`, `Storage` with custom cache folder, `Shortcuts`, `About`), custom `.onnx` model inspection/swapping, and Markdown release update notifications.
- **Instant Offline-First Startup:** Locally bundled fonts and icons, instant pre-React splash reveal, and code-split workspaces.
