# OrthoStyle: Manifold-Preserving Training-Free Style Transfer

Academic project page for **OrthoStyle: Manifold-Preserving Training-Free Style Transfer via Score-Orthogonal Guidance** (Oral Presentation at **SOICT 2026**).

Built with modern CSS and JavaScript, fast animations, and interactive charts. It can be hosted directly on **GitHub Pages** (100% static, client-side, zero server/GPU requirements).

---

## Key Features & Visual Interactions

- **Academic Theme (SOICT 2026 Oral)**:
  - **Scroll Progress Bar**: Shows reading progress at the top of the page.
  - **Sticky Navbar**: Clear navigation with active section highlighting.
  - **Hero Section**: Clean design with animated stat counters (`+25.8%`, `79.87`, `0 Steps`).
  - **Overview & Abstract**: Teaser figure followed by a clear summary of the paper.
  - **Interactive Analysis**:
    - **Style-Content Pareto Frontier**: Interactive scatter plot showing the balance between style fidelity and content preservation.
    - **Head-to-Head Gains vs. RB-Modulation**: Visual comparison bars highlighting improvements on the Stable Cascade backbone.
    - **Metric Comparator**: Filter across CSD, DINO, CLIP-I, LPIPS, and DCL with clear rankings.
    - **Pushforward Step Control**: Interactive view showing how parameter τ adjusts content preservation and style strength.
  - **Perceptual Superiority Breakdown**: Progress rings showing relative advantages over prior methods.
  - **Method Cards with Details**: Cards expand smoothly on hover or click to explain each module.
  - **Ablation Study**:
    - **a. Core Components Analysis**: Full-width table, Fig. 6, and notes for each module.
    - **b. Pushforward Step Threshold Controllability**: Table and slider explaining the effect of parameter τ.
    - **c. Effect of Prompt Conditioning Levels**: Comparison across null, object, and style prompts with Fig. 7.
  - **Interactive 15×15 Demo**: 225 precomputed results loaded instantly from `assets/results/`.
  - **Full-Screen Lightbox**: Click any image or demo result to zoom in high resolution.
  - **BibTeX One-Click Copy**: Fast citation copying with visual feedback.

---

## Directory Structure

```text
style_transfer_project_page/
├── index.html                           # Main project page (SOICT 2026 Oral)
├── style.css                            # Modern stylesheet
├── script.js                            # Interactive logic (demo, charts, lightbox, counters)
├── data.js                              # Content & style thumbnail definitions (15 x 15)
├── SOICT_2026_OthorStyle.pdf            # Research paper PDF
├── .nojekyll                            # GitHub Pages static configuration
├── .gitignore                           # Git ignore rules
└── assets/
    ├── favicon.svg                      # Page tab icon
    ├── content/                         # 15 content target images
    ├── style/                           # 15 style reference images
    ├── results/                         # 225 precomputed stylized results (webp)
    └── figures/                         # Optimized figures (pipeline, pareto, baselines, ablation)
```
