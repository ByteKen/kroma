## Summary

<!-- Describe the problem this solves in your own words and link the related issue (e.g. Closes #123). -->

## Changes & Rationale

<!-- Outline the key changes and why this approach was chosen. -->

## Verification

<!-- Note which automated checks and manual desktop scenarios (e.g. RAW/JPEG, zoom/pan, export, Windows/macOS) you ran. -->

- [ ] `npm --prefix kroma-web run typecheck`
- [ ] `npm --prefix kroma-web test`
- [ ] `npm --prefix kroma-web run build`
- [ ] `cargo test --workspace`
- [ ] `cargo clippy --workspace --all-targets -- -D warnings`

## Risks, Tradeoffs, or Limitations

<!-- Call out any performance, VRAM, sidecar compatibility, or platform caveats (or write "None"). -->

## AI Disclosure

<!-- If you used AI tools, briefly explain how they were used and how you reviewed/verified the output. Otherwise write "None". -->

- **Tool(s) used:**
- **Scope of assistance:**
- **Manual review & verification:**

## Checklist

- [ ] My changes are focused and avoid unrelated edits
- [ ] I understand and can maintain all code in this PR
- [ ] New/changed icons (if any) were regenerated via `npm --prefix kroma-web run icons:subset`
- [ ] No model binaries or temporary files are committed
