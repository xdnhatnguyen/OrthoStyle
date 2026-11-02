(() => {
  /* ═══════════════════════════════════════════════════════════════
     1. Interactive Demo (Precomputed Cache Lookup)
     ═══════════════════════════════════════════════════════════════ */
  const config = window.PROJECT_DEMO || {
    resultDirectory: "assets/results",
    resultExtensions: ["webp", "png", "jpg", "jpeg", "svg"],
    fakeLoadingMs: 700
  };

  let selectedContent = null;
  let selectedStyle = null;
  let generationToken = 0;

  const contentGrid = document.getElementById("content-grid");
  const styleGrid = document.getElementById("style-grid");
  const generateBtn = document.getElementById("generate-btn");
  const clearBtn = document.getElementById("clear-btn");
  const resultPlaceholder = document.getElementById("result-placeholder");
  const resultView = document.getElementById("result-view");
  const resultImage = document.getElementById("result-image");
  const loadingOverlay = document.getElementById("loading-overlay");
  const loadingTitle = document.getElementById("loading-title");
  const loadingSubtitle = document.getElementById("loading-subtitle");
  const progressBarDemo = document.getElementById("progress-bar-demo");
  const progressLabel = document.getElementById("progress-label");
  const statusTitle = document.getElementById("status-title");
  const statusDetail = document.getElementById("status-detail");
  const canvasWrapper = document.getElementById("canvas-wrapper");

  function renderGrid(items, grid, kind) {
    if (!grid || !items) return;
    items.forEach((item) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "thumb-card";
      button.dataset.kind = kind;
      button.dataset.id = item.id;
      button.setAttribute("aria-label", `${kind}: ${item.label}`);
      button.innerHTML = `<img src="${item.src}" alt="${item.label}" loading="lazy"><span class="thumb-label">${item.label}</span>`;
      button.addEventListener("click", () => selectItem(kind, item, button));
      grid.appendChild(button);
    });
  }

  function selectItem(kind, item, button) {
    const grid = kind === "content" ? contentGrid : styleGrid;
    grid.querySelectorAll(".thumb-card").forEach((el) => el.classList.remove("selected"));
    button.classList.add("selected");

    if (kind === "content") selectedContent = item;
    else selectedStyle = item;

    if (generateBtn) {
      generateBtn.disabled = !(selectedContent && selectedStyle);
    }

    if (resultView && !resultView.classList.contains("hidden")) {
      resultView.classList.add("hidden");
      resultPlaceholder.classList.remove("hidden");
      if (statusTitle) statusTitle.textContent = "Selection updated";
      if (statusDetail) statusDetail.textContent = "Click '✨ Generate Result' to view the precomputed stylized output.";
    }
  }

  function candidatePaths(contentId, styleId) {
    const stem = `${config.resultDirectory}/ortho_${contentId}_${styleId}`;
    return config.resultExtensions.map((ext) => `${stem}.${ext}`);
  }

  function findExistingImage(paths, index = 0) {
    return new Promise((resolve, reject) => {
      if (index >= paths.length) {
        reject(new Error("No matching precomputed image found."));
        return;
      }
      const probe = new Image();
      probe.onload = () => resolve(paths[index]);
      probe.onerror = () => findExistingImage(paths, index + 1).then(resolve).catch(reject);
      probe.src = paths[index];
    });
  }

  function preloadImage(src) {
    return new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => resolve(src);
      image.onerror = reject;
      image.src = src;
    });
  }

  function wait(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  function setDemoProgress(value) {
    const pct = Math.max(0, Math.min(100, value));
    if (progressBarDemo) progressBarDemo.style.width = `${pct}%`;
    if (progressLabel) progressLabel.textContent = `${Math.round(pct)}%`;
  }

  function runFakeProgress(ms, token) {
    setDemoProgress(0);
    const started = performance.now();

    return new Promise((resolve) => {
      function tick(now) {
        if (token !== generationToken) {
          resolve();
          return;
        }

        const elapsed = now - started;
        const ratio = Math.min(elapsed / ms, 1);
        const eased = 100 * (1 - Math.pow(1 - ratio, 1.6));
        const displayValue = ratio < 1 ? Math.min(eased, 96) : 100;
        setDemoProgress(displayValue);

        if (ratio >= 1) {
          resolve();
          return;
        }
        requestAnimationFrame(tick);
      }
      requestAnimationFrame(tick);
    });
  }

  async function generate() {
    if (!selectedContent || !selectedStyle) return;

    generationToken += 1;
    const token = generationToken;

    generateBtn.disabled = true;
    const origText = generateBtn.textContent;
    generateBtn.textContent = "Synthesizing…";
    clearBtn.disabled = true;

    if (loadingTitle) loadingTitle.textContent = "Synthesizing result…";
    if (loadingSubtitle) loadingSubtitle.textContent = `${selectedContent.label} × ${selectedStyle.label}`;
    if (loadingOverlay) loadingOverlay.classList.remove("hidden");

    try {
      const [path] = await Promise.all([
        (async () => {
          const found = await findExistingImage(candidatePaths(selectedContent.id, selectedStyle.id));
          await preloadImage(found);
          return found;
        })(),
        runFakeProgress(config.fakeLoadingMs || 700, token)
      ]);

      if (token !== generationToken) return;

      setDemoProgress(100);
      await wait(80);

      resultImage.src = path;
      if (canvasWrapper) {
        canvasWrapper.dataset.lb = path;
        canvasWrapper.dataset.lbCap = `OrthoStyle precomputed result: ${selectedContent.label} × ${selectedStyle.label}`;
      }
      resultPlaceholder.classList.add("hidden");
      resultView.classList.remove("hidden");
    } catch (error) {
      if (token !== generationToken) return;
      resultView.classList.add("hidden");
      resultPlaceholder.classList.remove("hidden");
      if (statusTitle) statusTitle.textContent = "Result not found";
      if (statusDetail) statusDetail.textContent = `ortho_${selectedContent.id}_${selectedStyle.id}.[webp/png] is missing.`;
    } finally {
      if (token !== generationToken) return;
      await wait(100);
      if (loadingOverlay) loadingOverlay.classList.add("hidden");
      generateBtn.textContent = origText || "✨ Generate Result";
      generateBtn.disabled = !(selectedContent && selectedStyle);
      clearBtn.disabled = false;
    }
  }

  function clearAll() {
    generationToken += 1;
    selectedContent = null;
    selectedStyle = null;
    document.querySelectorAll(".thumb-card.selected").forEach((el) => el.classList.remove("selected"));
    if (generateBtn) generateBtn.disabled = true;
    if (clearBtn) clearBtn.disabled = false;
    if (loadingOverlay) loadingOverlay.classList.add("hidden");
    setDemoProgress(0);
    if (resultView) resultView.classList.add("hidden");
    if (resultPlaceholder) {
      resultPlaceholder.classList.remove("hidden");
      if (statusTitle) statusTitle.textContent = "Choose a content and style above";
      if (statusDetail) statusDetail.textContent = "Click any thumbnail from Step 1 and Step 2, then hit Generate Result.";
    }
  }

  if (config.content && contentGrid) renderGrid(config.content, contentGrid, "content");
  if (config.style && styleGrid) renderGrid(config.style, styleGrid, "style");
  if (generateBtn) generateBtn.addEventListener("click", generate);
  if (clearBtn) clearBtn.addEventListener("click", clearAll);

  /* ═══════════════════════════════════════════════════════════════
     2. Top Scroll Progress Bar & Sticky Navbar
     ═══════════════════════════════════════════════════════════════ */
  const topProgressBar = document.getElementById("progress-bar");
  const navbar = document.getElementById("navbar");

  window.addEventListener("scroll", () => {
    const scrollMax = document.documentElement.scrollHeight - window.innerHeight;
    const scrollPct = scrollMax > 0 ? (window.scrollY / scrollMax) * 100 : 0;
    if (topProgressBar) topProgressBar.style.width = scrollPct + "%";

    if (navbar) {
      const heroEl = document.querySelector(".hero");
      const heroHeight = heroEl ? heroEl.offsetHeight : 400;
      navbar.classList.toggle("visible", window.scrollY > heroHeight - 90);
    }
  }, { passive: true });

  /* Active nav link highlight on scroll */
  const navSections = ["overview", "demo", "abstract", "method", "quantitative", "advantage", "analysis", "results", "ablation", "bibtex"];
  window.addEventListener("scroll", () => {
    const curY = window.scrollY + 120;
    let currentId = navSections[0];
    navSections.forEach((id) => {
      const el = document.getElementById(id);
      if (el && el.offsetTop <= curY) {
        currentId = id;
      }
    });
    document.querySelectorAll(".nav-links a").forEach((a) => {
      const href = a.getAttribute("href");
      a.classList.toggle("active", href === "#" + currentId);
    });
  }, { passive: true });

  /* ═══════════════════════════════════════════════════════════════
     3. Scroll Reveal Animations (.reveal, .reveal-left, .reveal-right)
     ═══════════════════════════════════════════════════════════════ */
  const revealObs = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("up");
        revealObs.unobserve(entry.target);
      }
    });
  }, { threshold: 0.08 });

  document.querySelectorAll(".reveal, .reveal-left, .reveal-right").forEach((el) => {
    revealObs.observe(el);
  });

  /* ═══════════════════════════════════════════════════════════════
     4. Number Count-up Animation (.count-up)
     ═══════════════════════════════════════════════════════════════ */
  function animateNumber(el) {
    const target = parseFloat(el.dataset.target);
    if (isNaN(target)) return;
    const prefix = el.dataset.prefix || "";
    const suffix = el.dataset.suffix || "";
    const decimals = parseInt(el.dataset.decimals || "0", 10);
    const duration = 1200;
    const start = performance.now();

    function tick(now) {
      const progress = Math.min((now - start) / duration, 1);
      const ease = 1 - Math.pow(1 - progress, 3);
      const cur = target * ease;
      el.textContent = prefix + cur.toFixed(decimals) + suffix;
      if (progress < 1) {
        requestAnimationFrame(tick);
      } else {
        el.textContent = prefix + target.toFixed(decimals) + suffix;
      }
    }
    requestAnimationFrame(tick);
  }

  const countObs = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.querySelectorAll(".count-up").forEach(animateNumber);
        countObs.unobserve(entry.target);
      }
    });
  }, { threshold: 0.3 });

  document.querySelectorAll(".hero-stats, .results-strip, .study-grid").forEach((el) => {
    countObs.observe(el);
  });

  /* Circular Progress Ring Animation (.study-grid) */
  const ringObs = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (e.isIntersecting) {
        e.target.querySelectorAll(".ring-fg").forEach((ring) => {
          const pct = parseFloat(ring.dataset.pct) / 100;
          const circ = 2 * Math.PI * 35; // r=35
          ring.style.strokeDashoffset = circ * (1 - pct);
        });
        ringObs.unobserve(e.target);
      }
    });
  }, { threshold: 0.25 });

  document.querySelectorAll(".study-grid").forEach((el) => ringObs.observe(el));

  /* ═══════════════════════════════════════════════════════════════
     5. Method Cards Expand ("Bung Chữ" on click / toggle)
     ═══════════════════════════════════════════════════════════════ */
  document.querySelectorAll(".mcard[data-expand]").forEach((card) => {
    card.addEventListener("click", () => {
      card.classList.toggle("expanded");
    });
  });

  /* ═══════════════════════════════════════════════════════════════
     6. Quantitative Benchmark Table Tabs
     ═══════════════════════════════════════════════════════════════ */
  const tableTabs = document.querySelectorAll(".table-tab[data-group]");
  const tableRows = document.querySelectorAll("#quant-table-wrap tbody tr");

  tableTabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tableTabs.forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");

      const group = tab.dataset.group;
      tableRows.forEach((row) => {
        if (group === "all") {
          row.style.display = "";
        } else if (group === "stylization") {
          row.style.display = row.classList.contains("tr-group-stylization") ? "" : "none";
        } else if (group === "oc") {
          row.style.display = row.classList.contains("tr-group-oc") ? "" : "none";
        }
      });
    });
  });

  /* ═══════════════════════════════════════════════════════════════
     7. Full-Screen Lightbox Modal
     ═══════════════════════════════════════════════════════════════ */
  const lightbox = document.getElementById("lightbox");
  const lbImg = document.getElementById("lb-img");
  const lbCap = document.getElementById("lb-cap");
  const lbClose = document.getElementById("lb-close");
  const lbPrev = document.getElementById("lb-prev");
  const lbNext = document.getElementById("lb-next");

  const lbItems = [];
  let lbCurrentIdx = 0;

  function refreshLightboxRegistry() {
    lbItems.length = 0;
    document.querySelectorAll("[data-lb]").forEach((el) => {
      const src = el.dataset.lb || el.getAttribute("src");
      const cap = el.dataset.lbCap || el.getAttribute("alt") || "";
      if (src) {
        lbItems.push({ el, src, cap });
        el.style.cursor = "zoom-in";
        el.onclick = (e) => {
          e.preventDefault();
          e.stopPropagation();
          openLightbox(src, cap);
        };
      }
    });
  }

  function openLightbox(src, cap) {
    if (!lightbox || !lbImg) return;
    lbCurrentIdx = lbItems.findIndex((item) => item.src === src);
    if (lbCurrentIdx === -1) lbCurrentIdx = 0;
    lbImg.src = src;
    if (lbCap) lbCap.textContent = cap || "";
    lightbox.classList.add("open");
    document.body.style.overflow = "hidden";
  }

  function closeLightbox() {
    if (!lightbox) return;
    lightbox.classList.remove("open");
    document.body.style.overflow = "";
  }

  function navigateLightbox(dir) {
    if (!lbItems.length) return;
    lbCurrentIdx = (lbCurrentIdx + dir + lbItems.length) % lbItems.length;
    const item = lbItems[lbCurrentIdx];
    if (item && lbImg) {
      lbImg.src = item.src;
      if (lbCap) lbCap.textContent = item.cap || "";
    }
  }

  if (lbClose) lbClose.addEventListener("click", closeLightbox);
  if (lbPrev) lbPrev.addEventListener("click", (e) => { e.stopPropagation(); navigateLightbox(-1); });
  if (lbNext) lbNext.addEventListener("click", (e) => { e.stopPropagation(); navigateLightbox(1); });

  if (lightbox) {
    lightbox.addEventListener("click", (e) => {
      if (e.target === lightbox) closeLightbox();
    });
  }

  window.addEventListener("keydown", (e) => {
    if (!lightbox || !lightbox.classList.contains("open")) return;
    if (e.key === "Escape") closeLightbox();
    else if (e.key === "ArrowLeft") navigateLightbox(-1);
    else if (e.key === "ArrowRight") navigateLightbox(1);
  });

  refreshLightboxRegistry();

  /* ═══════════════════════════════════════════════════════════════
     8. Copy BibTeX Citation Button
     ═══════════════════════════════════════════════════════════════ */
  const copyBtn = document.getElementById("copy-bib-btn");
  const copyBtnText = document.getElementById("copy-btn-text");

  if (copyBtn) {
    copyBtn.addEventListener("click", async () => {
      const bibText = `@inproceedings{nguyen2026orthostyle,
  title     = {OrthoStyle: Manifold-Preserving Training-Free Style Transfer via Score-Orthogonal Guidance},
  author    = {Nguyen, Minh Nhat and Pham, Dang Quang and Vo, Thien Phuc and Dang, Phuc Loc and Le, Duc Khoan},
  booktitle = {The 15th International Symposium on Information and Communication Technology (SOICT)},
  note      = {Oral Presentation},
  year      = {2026}
}`;
      try {
        await navigator.clipboard.writeText(bibText);
        copyBtn.classList.add("copied");
        if (copyBtnText) copyBtnText.textContent = "✓ Copied!";
        setTimeout(() => {
          copyBtn.classList.remove("copied");
          if (copyBtnText) copyBtnText.textContent = "Copy BibTeX";
        }, 2200);
      } catch (err) {
        if (copyBtnText) copyBtnText.textContent = "Press Ctrl+C";
      }
    });
  }

  /* ═══════════════════════════════════════════════════════════════
     9. Visual & Trade-off Analysis: Interactive Visualizations
     ═══════════════════════════════════════════════════════════════ */

  // Helper function to position tooltip inside a relative container
  function bindHoverTooltip(containerId, itemSelector, tooltipId) {
    const container = document.getElementById(containerId);
    const tooltip = document.getElementById(tooltipId);
    if (!container || !tooltip) return;

    container.querySelectorAll(itemSelector).forEach((item) => {
      item.addEventListener("mouseenter", () => {
        const info = item.dataset.info;
        if (!info) return;
        tooltip.textContent = info;
        const itemRect = item.getBoundingClientRect();
        const contRect = container.getBoundingClientRect();
        const x = itemRect.left + itemRect.width / 2 - contRect.left;
        const y = itemRect.top - contRect.top;
        tooltip.style.left = `${x}px`;
        tooltip.style.top = `${y}px`;
        tooltip.classList.add("visible");
      });

      item.addEventListener("mousemove", (e) => {
        const contRect = container.getBoundingClientRect();
        const x = e.clientX - contRect.left;
        const y = e.clientY - contRect.top - 14;
        tooltip.style.left = `${x}px`;
        tooltip.style.top = `${y}px`;
      });

      item.addEventListener("mouseleave", () => {
        tooltip.classList.remove("visible");
      });
    });
  }

  // 9.2 Tooltips for Pareto Scatter Points and Gain Rows
  bindHoverTooltip("interactive-pareto", ".chart-pt", "tooltip-pareto");
  bindHoverTooltip("interactive-rbm", ".gain-row", "tooltip-rbm");

  // 9.3 Table 1 Interactive Metric Comparator (Card 3)
  const TABLE1_DATA = {
    clip: {
      name: "CLIP-I (Higher is better ↑)",
      stylization: [
        { name: "OrthoStyle", val: 79.87, ours: true, best: true },
        { name: "StyleID", val: 77.32, second: true },
        { name: "AttenST", val: 77.17 },
        { name: "DiffuseIT", val: 68.28 },
        { name: "InstantStyle", val: 68.27 }
      ],
      oc: [
        { name: "ITOC", val: 86.57, best: true },
        { name: "RB-Modulation", val: 81.69, second: true },
        { name: "OrthoStyle", val: 79.87, ours: true }
      ]
    },
    csd: {
      name: "CSD (Higher is better ↑)",
      stylization: [
        { name: "InstantStyle", val: 47.67, best: true },
        { name: "StyleID", val: 45.11, second: true },
        { name: "AttenST", val: 41.44 },
        { name: "OrthoStyle", val: 39.49, ours: true },
        { name: "DiffuseIT", val: 31.90 }
      ],
      oc: [
        { name: "OrthoStyle", val: 39.49, ours: true, best: true },
        { name: "RB-Modulation", val: 31.38, second: true },
        { name: "ITOC", val: 14.62 }
      ]
    },
    dino: {
      name: "DINO (Higher is better ↑)",
      stylization: [
        { name: "DiffuseIT", val: 32.11, best: true },
        { name: "InstantStyle", val: 27.80, second: true },
        { name: "StyleID", val: 26.95 },
        { name: "AttenST", val: 24.80 },
        { name: "OrthoStyle", val: 23.86, ours: true }
      ],
      oc: [
        { name: "OrthoStyle", val: 23.86, ours: true, best: true },
        { name: "RB-Modulation", val: 19.13, second: true },
        { name: "ITOC", val: 16.04 }
      ]
    },
    lpips: {
      name: "LPIPS (Lower is better ↓)",
      stylization: [
        { name: "AttenST", val: 43.47, best: true },
        { name: "StyleID", val: 45.97, second: true },
        { name: "DiffuseIT", val: 54.43 },
        { name: "OrthoStyle", val: 56.22, ours: true },
        { name: "InstantStyle", val: 61.55 }
      ],
      oc: [
        { name: "ITOC", val: 33.86, best: true },
        { name: "OrthoStyle", val: 56.22, ours: true, second: true },
        { name: "RB-Modulation", val: 68.09 }
      ],
      lowerBetter: true
    },
    dcl: {
      name: "DCL (Lower is better ↓)",
      stylization: [
        { name: "AttenST", val: 26.07, best: true },
        { name: "OrthoStyle", val: 26.41, ours: true, second: true },
        { name: "StyleID", val: 27.05 },
        { name: "InstantStyle", val: 36.49 },
        { name: "DiffuseIT", val: 42.69 }
      ],
      oc: [
        { name: "ITOC", val: 19.39, best: true },
        { name: "OrthoStyle", val: 26.41, ours: true, second: true },
        { name: "RB-Modulation", val: 28.81 }
      ],
      lowerBetter: true
    }
  };

  const metricDisplay = document.getElementById("metric-bars-display");
  function renderMetricComparator(metricKey) {
    if (!metricDisplay || !TABLE1_DATA[metricKey]) return;
    const data = TABLE1_DATA[metricKey];

    // Compute dynamic maximum value from the data logic
    const allVals = [...data.stylization, ...data.oc].map((i) => i.val);
    const maxVal = Math.max(...allVals);

    function renderGroupRows(items) {
      return items.map((item) => {
        const pct = Math.min(100, Math.max(4, (item.val / maxVal) * 100));
        let fillClass = "mbar-fill";
        if (item.ours) fillClass += " ours";
        else if (item.best) fillClass += " best";
        else if (item.second) fillClass += " second";

        let badgeTag = "";
        if (item.best) badgeTag = `<span class="mbar-badge-tag tag-best">★ Best</span>`;
        else if (item.second) badgeTag = `<span class="mbar-badge-tag tag-second">#2</span>`;

        const info = `${item.name}: ${item.val.toFixed(2)} on ${data.name}`;
        const valFormatted = item.best
          ? `<strong>${item.val.toFixed(2)}</strong>`
          : (item.second ? `<u>${item.val.toFixed(2)}</u>` : item.val.toFixed(2));

        return `
          <div class="mbar-row ${item.ours ? 'active' : ''}" data-info="${info}">
            <div class="mbar-label">
              <span class="mbar-name ${item.ours ? 'ours' : ''}">${item.name}</span>
              ${badgeTag}
            </div>
            <div class="mbar-track">
              <div class="${fillClass}" style="width: ${pct}%;"></div>
            </div>
            <span class="mbar-val ${item.ours ? 'ours' : ''}">${valFormatted}</span>
          </div>
        `;
      }).join("");
    }

    metricDisplay.innerHTML = `
      <div class="mbar-group">
        <div class="mbar-group-title">Stylization Methods</div>
        ${renderGroupRows(data.stylization)}
      </div>
      <div class="mbar-group" style="margin-top: 6px;">
        <div class="mbar-group-title">OC-Based Methods</div>
        ${renderGroupRows(data.oc)}
      </div>
    `;

    bindHoverTooltip("interactive-metric", ".mbar-row", "tooltip-metric");
  }

  document.querySelectorAll(".comp-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".comp-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      renderMetricComparator(btn.dataset.metric);
    });
  });
  renderMetricComparator("clip");

  // 9.4 Table 2(b) Controllability via Pushforward Threshold τ (Card 4)
  const TAU_DATA = {
    1: {
      label: "τ = 1 (Default OrthoStyle Setting)",
      csd: 45.14,
      dino: 27.34,
      clip: 75.90,
      lpips: 58.55,
      dcl: 27.27,
      desc: "Maximum artistic stylization intensity with vibrant textures."
    },
    2: {
      label: "τ = 2",
      csd: 41.57,
      dino: 25.51,
      clip: 76.50,
      lpips: 56.61,
      dcl: 25.50,
      desc: "Balanced transition with improved structural retention."
    },
    3: {
      label: "τ = 3",
      csd: 40.40,
      dino: 24.54,
      clip: 78.44,
      lpips: 55.51,
      dcl: 23.52,
      desc: "Enhanced content geometry and reduced background distortion."
    },
    4: {
      label: "τ = 4",
      csd: 37.07,
      dino: 23.27,
      clip: 79.40,
      lpips: 54.22,
      dcl: 21.65,
      desc: "Highest content preservation (+3.50 CLIP-I) and lowest leakage (-5.62 DCL)."
    }
  };

  const tauDisplay = document.getElementById("tau-bars-display");
  function renderTauControllability(tauKey) {
    if (!tauDisplay || !TAU_DATA[tauKey]) return;
    const t = TAU_DATA[tauKey];

    const metrics = [
      { name: "CLIP-I Content ↑", val: t.clip, color: "linear-gradient(90deg, #635bff, #a855f7)", info: `CLIP-I: ${t.clip} (higher is better)` },
      { name: "CSD Style Sim. ↑", val: t.csd, color: "linear-gradient(90deg, #7c3aed, #c084fc)", info: `CSD: ${t.csd} (higher is better)` },
      { name: "DINO Alignment ↑", val: t.dino, color: "linear-gradient(90deg, #9333ea, #e879f9)", info: `DINO: ${t.dino} (higher is better)` },
      { name: "LPIPS Distance ↓", val: t.lpips, color: "linear-gradient(90deg, #059669, #10b981)", info: `LPIPS: ${t.lpips} (lower is better)` },
      { name: "DCL Leakage ↓", val: t.dcl, color: "linear-gradient(90deg, #0d9488, #14b8a6)", info: `DCL: ${t.dcl} (lower is better)` }
    ];

    const rowsHtml = metrics.map((m) => {
      const pct = Math.min(100, Math.max(4, m.val));
      return `
        <div class="mbar-row" data-info="${m.info}">
          <div class="mbar-label">
            <span class="mbar-name">${m.name}</span>
          </div>
          <div class="mbar-track">
            <div class="mbar-fill" style="width: ${pct}%; background: ${m.color};"></div>
          </div>
          <span class="mbar-val">${m.val.toFixed(2)}</span>
        </div>
      `;
    }).join("");

    tauDisplay.innerHTML = `
      <div style="font-size: 12.5px; font-weight: 700; color: var(--primary); margin-bottom: 4px;">
        ${t.label}: <span style="font-weight: normal; color: var(--text-muted); font-size: 12px;">${t.desc}</span>
      </div>
      <div class="mbar-group">
        ${rowsHtml}
      </div>
    `;

    bindHoverTooltip("interactive-tau", ".mbar-row", "tooltip-tau");
  }

  document.querySelectorAll(".tau-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tau-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      renderTauControllability(btn.dataset.tau);
    });
  });
  renderTauControllability("1");

})();
