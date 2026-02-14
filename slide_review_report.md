# Slide Review Report

**Deck:** `slides/index.html` + `slides/slides.md` (Reveal.js 4.5.0, external Markdown)
**Date:** 2026-02-14
**Total slides:** 29 (28 horizontal + 1 vertical sub-slide after Barrowman split)
**Total equations:** 10 display-math blocks (`$$`), 5 inline-math spans (`$...$`)
**Total images:** 12 (+ 2 videos)

## Summary

| Category | Issues found | Issues fixed |
|----------|-------------|-------------|
| Equation rendering | 4 | 4 |
| Image rendering | 7 | 7 |
| Slide overflow / readability | 4 | 4 |
| Deployment | 3 | 3 |
| **Total** | **18** | **18** |

All 18 issues have been resolved.

---

## 1. Equation Rendering

### EQ-1 (HIGH) — MathJax/KaTeX plugin conflict -- FIXED

**File:** `slides/index.html`
**Was:** Dead `math` config block referencing MathJax 3, plus `.mjx-chtml` CSS rule targeting a MathJax class that KaTeX never generates.
**Fix applied:**
- Replaced `math: { mathjax: ..., config: ... }` with `katex: { version: '0.16.9' }` to pin KaTeX version
- Replaced `.mjx-chtml` CSS selector with `.katex` (the actual KaTeX class)
- Removed unused `.university-bg` CSS rule with placeholder `path/to/your/logo.png`

### EQ-2 (MEDIUM) — Wind torque equation oversimplified vs code -- FIXED

**File:** `slides/slides.md`, slide "Why Classical Fails on J800"
**Was:** `$$\tau_{wind} \propto \sin(\theta_{wind} - \psi_{roll})$$`
**Fix applied:** Changed to `$$\tau_{wind} \propto \sin\!\bigl(2(\theta_{wind} - \phi_{roll})\bigr)$$` to match the 4-fin `sin(2*gamma)` in `wind_model.py`. Updated bullet to note "Torque oscillates at **twice** the spin frequency (4-fin symmetry)". Expanded speaker notes to explain the factor of 2.

### EQ-3 (MEDIUM) — Inconsistent notation for roll angle -- FIXED

**File:** `slides/slides.md`, slide "Why Classical Fails on J800"
**Was:** Roll angle denoted `\psi_{roll}` (psi is conventionally yaw in aerospace).
**Fix applied:** Changed to `\phi_{roll}` (phi = roll, theta = pitch, psi = yaw per aerospace convention).

### EQ-4 (LOW) — Barrowman fin CN formula not implemented in code -- FIXED

**File:** `slides/slides.md`, slide "Stability: The Barrowman Equations" speaker notes
**Was:** No mention that Barrowman equations are a pre-flight design tool, not used in the simulation.
**Fix applied:** Added clarification to speaker notes: "These are used for pre-flight airframe design; the flight simulation assumes an already-stable airframe."

---

## 2. Image Rendering

### IMG-1 (HIGH) — Aspect ratio distortion: `hiking_with_rockets.png` -- FIXED

**File:** `slides/slides.md`
**Was:** `width="400" height="480"` — 37.5% distortion (landscape source forced into portrait frame)
**Fix applied:** Removed `height` attribute. Now `width="400"` with browser auto-height.

### IMG-2 (HIGH) — Aspect ratio distortion: `ARGOS_website_screenshot.png` -- FIXED

**File:** `slides/slides.md`
**Was:** `width="300" height="200"` — 43.5% distortion (near-square source stretched to 3:2)
**Fix applied:** Removed `height` attribute. Now `width="300"` with auto-height.

### IMG-3 (HIGH) — Aspect ratio distortion: `EPFL_website_screenshot.png` -- FIXED

**File:** `slides/slides.md`
**Was:** `width="300" height="200"` — 21% distortion
**Fix applied:** Removed `height` attribute. Now `width="300"` with auto-height.

### IMG-4 (HIGH) — Placeholder logo path in CSS -- FIXED

**File:** `slides/index.html`
**Was:** `.university-bg` CSS rule with `url('path/to/your/logo.png')` — dead template code.
**Fix applied:** Removed the entire `.university-bg` CSS rule.

### IMG-5 (MEDIUM) — Aspect ratio distortion: `frenzy_off_axis.JPG` -- FIXED

**File:** `slides/slides.md`
**Was:** `width="500" height="400"` — 16.7% distortion
**Fix applied:** Removed `height` attribute. Now `width="500"` with auto-height.
**Note:** Source resolution (792px) is borderline for 500px display on HiDPI. Replace with higher-res source if available.

### IMG-6 (MEDIUM) — Aspect ratio distortion: `rocket_stability.gif` -- FIXED

**File:** `slides/slides.md`
**Was:** `width="400" height="400"` — 14.3% distortion (portrait GIF forced to square)
**Fix applied:** Changed to `width="350"` with auto-height (~400px), fitting the two-column layout.

### IMG-7 (LOW) — Videos in square containers -- FIXED

**File:** `slides/slides.md`
**Was:** Both portrait videos (9:16) displayed in `360x360` square containers, causing letterboxing.
**Fix applied:** Changed both to `width="270" height="480"` to match the 9:16 portrait aspect ratio.

---

## 3. Slide Overflow / Readability

### OVF-1 (HIGH) — Barrowman Equations slide overflow -- FIXED

**File:** `slides/slides.md`
**Was:** Single slide with H2 + paragraph + CP formula + CN table (including complex fin formula). Estimated ~700-750px, overflowing the 620px usable area.
**Fix applied:** Split into two slides using `----` (vertical sub-slide):
1. "Stability: The Barrowman Equations" — CP formula + brief explanation
2. "Component Normal Force Coefficients" — CN table + summary line

### OVF-2 (MEDIUM) — Gain Scheduling slide tight fit -- FIXED

**File:** `slides/slides.md`
**Was:** 3 bullet points + emphasis line below the equation. Estimated ~580-620px.
**Fix applied:** Removed the "Gains clamped to [0.5x, 5x]" bullet (moved to speaker notes). Now 2 bullets + emphasis line, fitting comfortably.

### OVF-3 (MEDIUM) — Control Hardware slide width -- FIXED

**File:** `slides/slides.md`
**Was:** Center column `flex: 1.2` with `width="340"` image, side columns with 4 bullets each. Total width ~1000px.
**Fix applied:**
- Changed center column from `flex: 1.2` to `flex: 1`
- Reduced image from `width="340" height="255"` to `width="300"` (auto-height, no distortion)
- Reduced each side column from 4 to 3 bullets (moved "25% chord x 50% span" and "~0.15 deg/s RMS noise" to speaker notes)

### OVF-4 (MEDIUM) — RL Formulation slide height -- FIXED

**File:** `slides/slides.md`
**Was:** Left column had 4 bullet sub-items for observations. Estimated ~600px.
**Fix applied:** Condensed the 4 observation bullets into 3 compact lines:
- "Altitude, velocity, roll angle/rate/accel"
- "$q$, time, thrust fraction"
- "Previous action, shake metric"

---

## 4. Deployment

### DEP-1 (HIGH) — Dead MathJax configuration -- FIXED

See EQ-1. Replaced with `katex: { version: '0.16.9' }`.

### DEP-2 (MEDIUM) — Unpinned Mermaid version -- FIXED

**File:** `slides/index.html`
**Was:** `mermaid@11` (resolves to latest 11.x)
**Fix applied:** Pinned to `mermaid@11.4.1`.

### DEP-3 (MEDIUM) — KaTeX version unpinned -- FIXED

**File:** `slides/index.html`
**Was:** No `katex` config block; KaTeX loaded as latest from CDN.
**Fix applied:** Added `katex: { version: '0.16.9' }` in `Reveal.initialize()`.

---

## Non-Issues (Verified OK)

- **All 14 content asset paths** in `slides.md` are relative and resolve to existing files.
- **HTML is well-formed** — doctype, lang, charset, viewport all present.
- **Reveal.js CSS/JS dependencies** are loaded from a pinned CDN version (4.5.0).
- **All LaTeX commands** used in equations are supported by KaTeX.
- **The `\_` escape pattern** for subscripts in Markdown is the correct workaround for Reveal.js Markdown + KaTeX and renders correctly.
- **Font sizes** are adequate: base is 32px, `.small` is 0.7em (22.4px), code is 0.55em (17.6px). No text below the flagging thresholds.
- **Fragments** are used correctly on the Objectives slide (slide 3) for incremental reveal.
- **Speaker notes** are present on nearly every content slide (good practice).
- **Equations: Newton's 2nd law, dynamic pressure, PID, gain scheduling, residual RL** — all accurately match the code implementations.
- **`control_tabs.jpeg`** is 2.8 MB at 4000x3000 for a 300px display — oversized but not a correctness issue. Could be resized for faster loading.
