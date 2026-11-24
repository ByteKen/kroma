# Contributing to Kroma

Thank you for your interest in improving **Kroma**. Focused, well-tested pull requests that respect Kroma's non-destructive darkroom design and local-first architecture are always welcome.

## Start With an Issue

Before starting substantial work, open a **Bug Report** or **Feature Request** and confirm the approach with a maintainer. Discussing scope and design up front prevents wasted effort on changes that conflict with ongoing engine or UI work.

Good starting points for first-time contributors:
- Scoped bug fixes with a focused unit test
- RAW/EXIF edge-case tests in Rust
- Accessibility, shortcut, or UI ergonomics refinements

Large, unannounced refactors or multi-feature PRs may be closed without review.

## Architecture Overview

Kroma is organized around Clean Architecture boundaries with strict inward dependencies:

- `internal/domain/` — Pure Rust domain models and invariants (`DevelopRecipe`, masking, identity clustering, slate parsing). Zero framework, Tauri, filesystem, or GPU imports.
- `internal/application/` — Focused Rust use cases (`cluster_shoot`, `organize_shoot`, `batch_render`) depending only on domain types and traits.
- `internal/infrastructure/` — Concrete adapters: RAW/image decoding (`rawler`, `image`), 32-bit Rayon CPU and `wgpu` compute develop engines, `.kroma.json` sidecar storage, and local ONNX Runtime engines (YuNet, ArcFace, PP-OCRv5, RMBG-1.4).
- `kroma-tau/` — Thin Tauri v2 desktop bridge (`commands.rs`, `dto.rs`, `state.rs`, `model_manager.rs`). Translates IPC calls without owning core business rules.
- `shaders/develop/` — Unified WGSL shaders (`grade_pass.wgsl`, `mask_pass.wgsl`, `spatial_pass.wgsl`, `present.wgsl`) shared between live WebGPU preview and Rust `wgpu` export.
- `kroma-web/` — React 19 + TypeScript + Tailwind CSS v4 frontend, layered into `domain/`, `application/`, `adapters/`, and `features/`.
- `tests/web/` — Vitest unit test suite covering frontend domain math, stores, and UI helpers.

### Key Invariants

1. **Strictly non-destructive:** Original camera RAW and image files must never be mutated. All edits and grouping metadata live in `.kroma.json` sidecars.
2. **Cross-engine develop parity:** Any math change in `color_math.rs`, `pixel_engine.rs`, or `mask_engine.rs` must stay aligned with `shaders/develop/*.wgsl`, `WebGlDevelopCanvas.tsx`, and `developShaderMath.ts` (verified by `wgpu_golden_parity_matches_rayon_cpu_across_full_develop_pipeline`).
3. **Offline-first & privacy-preserving:** Runtime workflows must never send user photos or metadata over the network.
4. **HTML5 drag-and-drop compatibility:** Keep `"dragDropEnabled": false` in `kroma-tau/tauri.conf.json` so Face Organizer drag-and-drop remains functional.

## Generated & Managed Assets

- **Material Symbols icon font:** Kroma bundles a compact offline subsetted WOFF2 at `kroma-web/src/assets/material-symbols-outlined.woff2` with its manifest at `kroma-web/src/assets/material-symbols-icons.json`. Whenever you add a new icon name in `kroma-web/src`, regenerate the subset instead of editing those files by hand:
  ```bash
  npm --prefix kroma-web run icons:subset
  ```
  A Vitest guard (`tests/web/assets/materialSymbolsManifest.test.ts`) automatically fails if any referenced icon is missing.
- **AI model weights:** Never commit `.onnx`, `.bin`, `.pt`, or other large model binaries to Git. Model downloads are managed exclusively through `models/models.json`.

## Coding Conventions

- Keep changes small, direct, and scoped to a single issue or capability.
- Match existing formatting and naming conventions; avoid unrelated reformatting or churn.
- Keep comments focused on invariants, math rationale, or GPU/IPC contracts rather than narrating obvious lines.
- Rust code uses Edition 2024 and must pass Clippy with `-D warnings`.
- TypeScript code must pass strict `tsc -b --noEmit` with zero errors.

## AI-Assisted Contributions

Using AI coding assistants is allowed, provided you remain fully accountable for the submitted code.

> [!IMPORTANT]
> Only submit code you thoroughly understand, have tested locally, and can debug yourself. Maintainers will not rewrite or troubleshoot unverified AI-generated diffs.

If AI tools assisted with research, implementation, tests, or debugging, include a short **AI Disclosure** section in your pull request description noting which tools you used, what they helped with, and how you manually verified the result.

## Pull Request Descriptions

Write your PR summary in your own words rather than pasting an automated diff recap. Clearly cover:

- **Problem:** What bug or workflow gap this addresses (and link the issue).
- **Changes & Rationale:** What changed and why this design was chosen.
- **Verification:** Commands run and manual desktop testing performed (RAW/JPEG, Windows/macOS, WebGPU/WebGL2 where relevant).
- **Tradeoffs or Limitations:** Any known edge cases, memory/VRAM considerations, or follow-up items.

## Local Development & Verification

### Branch Workflow

```bash
git checkout main
git pull origin main
git checkout -b fix/your-short-topic
```

Before opening a pull request, rebase on the latest `main`:

```bash
git fetch origin
git rebase origin/main
```

### Required Local Checks

Run the full verification suite before submitting a PR:

```bash
# Frontend typecheck, unit tests, and production bundle
npm --prefix kroma-web run typecheck
npm --prefix kroma-web test
npm --prefix kroma-web run build

# Rust workspace tests and zero-warning Clippy
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings

# Repository hygiene & manifest validation
node scripts/release-check.mjs --check-manifest --check-hygiene
```

## Pre-PR Checklist

- [ ] Linked an approved issue (for new features or architectural changes)
- [ ] Tested locally in the desktop app (`npm --prefix kroma-web run tauri:dev`)
- [ ] Added or updated unit tests for changed behavior
- [ ] Frontend `typecheck`, `test`, and `build` pass cleanly
- [ ] Rust `cargo test --workspace` and `cargo clippy --workspace --all-targets -- -D warnings` pass with zero warnings
- [ ] Regenerated subsetted icons if any new Material Symbols were added
- [ ] Wrote the PR description in my own words and disclosed any AI assistance
