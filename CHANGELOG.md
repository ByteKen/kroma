# Changelog

All notable changes to **Kroma** are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-11-22

### Added
- **32-Bit Wide-Gamut RAW & WGSL GPU Engine:** Unified WGSL compute pipeline running in Linear Rec.2020 on WebGPU (live studio with automatic WebGL2 fallback) and `wgpu` (DX12 on Windows, Metal on macOS) for batch export—supporting 29+ RAW formats, tone curves, HSL color mixer, 3-way color grading, 3D `.cube` LUTs, film grain, crop/level, local AI/manual masking, and non-destructive studio backdrop replacement.
- **Cross-Platform Desktop (Windows & macOS):** Native Tauri v2 builds for Windows (DirectML) and macOS Apple Silicon & Intel (CoreML) with platform-native window controls and keyboard shortcuts.
- **Face Organizer & Virtual Folders:** On-demand face clustering (YuNet + ArcFace R50 512-D), Time Assist, Ambiguous Review Queue, Manual Match Split View, nested virtual folders, and independent Develop & Organization undo/redo history.
- **Slate OCR:** Local PP-OCRv5 whiteboard scanning with smart high-resolution RAW fallback for automatic subject naming.
- **Selective Batch Export:** Export single photos, specific folders, or full shoots as graded JPEG, PNG, or 16-bit TIFF with optional original RAW pairing, custom naming patterns, slate archiving, and real-time output size estimation.
- **Manifest-Driven AI Bootstrapping:** Automatic verification and streaming SHA-256-verified model downloads on first launch via `models/models.json` (`models-v1` release assets).
- **Interactive Settings & Update Gate:** 6-tab Settings modal (`General`, `Performance`, `Models`, `Storage` with custom cache folder, `Shortcuts`, `About`), custom `.onnx` model inspection/swapping, and Markdown release update notifications.
- **Instant Offline-First Startup:** Locally bundled fonts and icons, instant pre-React splash reveal, and code-split workspaces.
