# Changelog

All notable changes to **Kroma** are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
