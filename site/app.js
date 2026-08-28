(() => {
  "use strict";

  const $ = (selector) => document.querySelector(selector);
  const canvas = $("#topology-canvas");
  const contextSelect = $("#context-select");
  const radiusRange = $("#radius-range");
  const radiusValue = $("#radius-value");
  const worldButton = $("#world-button");
  const playButton = $("#play-button");
  const worldReadout = $("#world-readout");
  const probabilityReadout = $("#probability-readout");
  const radiusReadout = $("#radius-readout");

  const palette = {
    background: "#091827",
    panel: "rgba(14, 34, 54, 0.88)",
    grid: "rgba(164, 202, 220, 0.12)",
    gridStrong: "rgba(164, 202, 220, 0.25)",
    free: "#2f8fa4",
    freeLight: "#69e4e8",
    unknown: "#ffc875",
    blocked: "#17293b",
    path: "#b9ef73",
    rose: "#ff8796",
    text: "#f4f7fb",
    muted: "#8496aa",
  };

  const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const state = {
    context: 0,
    radius: 0,
    doorwayOpen: true,
    playing: !prefersReducedMotion,
    lastWorldSwitch: performance.now(),
  };

  function eventProbabilities() {
    const prior = state.context === 0 ? 0.2 : 0.8;
    return [prior, prior * 0.72, prior * 0.34];
  }

  function updateReadout() {
    const probabilities = eventProbabilities();
    const sampleValid = state.doorwayOpen && state.radius <= 1;
    radiusValue.textContent = `r = ${state.radius}`;
    worldReadout.textContent = `doorway ${state.doorwayOpen ? "OPEN" : "BLOCKED"}`;
    worldReadout.style.color = state.doorwayOpen ? palette.path : palette.rose;
    probabilityReadout.textContent = `q ≈ ${probabilities[state.radius].toFixed(2)}`;
    probabilityReadout.style.color = palette.path;
    radiusReadout.textContent = sampleValid
      ? `support-valid at r = ${state.radius}`
      : `no valid sample path at r = ${state.radius}`;
    radiusReadout.style.color = sampleValid ? palette.path : palette.rose;
  }

  function setPlaying(playing) {
    state.playing = playing;
    playButton.textContent = playing ? "Pause animation" : "Resume animation";
    playButton.setAttribute("aria-pressed", String(playing));
    state.lastWorldSwitch = performance.now();
  }

  contextSelect.addEventListener("change", () => {
    state.context = Number(contextSelect.value);
    updateReadout();
  });
  radiusRange.addEventListener("input", () => {
    state.radius = Number(radiusRange.value);
    updateReadout();
  });
  worldButton.addEventListener("click", () => {
    state.doorwayOpen = !state.doorwayOpen;
    state.lastWorldSwitch = performance.now();
    updateReadout();
  });
  playButton.addEventListener("click", () => setPlaying(!state.playing));

  const ctx = canvas.getContext("2d");
  let width = 0;
  let height = 0;

  function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    width = rect.width;
    height = rect.height;
    canvas.width = Math.max(1, Math.round(width * dpr));
    canvas.height = Math.max(1, Math.round(height * dpr));
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function roundedRect(x, y, w, h, radius) {
    const r = Math.min(radius, w / 2, h / 2);
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  function drawLabel(text, x, y, color = palette.muted, size = 10, align = "left") {
    ctx.save();
    ctx.fillStyle = color;
    ctx.font = `600 ${size}px ui-monospace, SFMono-Regular, Menlo, monospace`;
    ctx.textAlign = align;
    ctx.textBaseline = "middle";
    ctx.fillText(text, x, y);
    ctx.restore();
  }

  function drawMap(panel, time) {
    const cols = 15;
    const rows = 11;
    const labelRoom = 34;
    const mapSize = Math.min(panel.w - 22, panel.h - labelRoom - 18);
    const cell = Math.min(mapSize / cols, (panel.h - labelRoom - 18) / rows);
    const mapW = cell * cols;
    const mapH = cell * rows;
    const ox = panel.x + Math.max(11, (panel.w - mapW) / 2);
    const oy = panel.y + labelRoom + Math.max(4, (panel.h - labelRoom - mapH) / 2);
    const wallCol = 7;
    const doorRows = [4, 5, 6];

    drawLabel("FIXED VISIBLE BEV  X", panel.x + 14, panel.y + 18, palette.freeLight, 9);
    drawLabel(state.doorwayOpen ? "SAMPLED WORLD: OPEN" : "SAMPLED WORLD: BLOCKED", panel.x + panel.w - 14, panel.y + 18, state.doorwayOpen ? palette.path : palette.rose, 9, "right");

    for (let row = 0; row < rows; row += 1) {
      for (let col = 0; col < cols; col += 1) {
        const x = ox + col * cell;
        const y = oy + row * cell;
        const border = cell * 0.08;
        const isOuterWall = row === 0 || row === rows - 1;
        const isHiddenWall = col === wallCol && row > 0 && row < rows - 1;
        const isDoor = isHiddenWall && doorRows.includes(row);
        let fill = palette.free;
        let alpha = 0.48;

        if (isOuterWall) {
          fill = palette.blocked;
          alpha = 1;
        } else if (isHiddenWall) {
          fill = isDoor && state.doorwayOpen ? palette.freeLight : palette.unknown;
          alpha = isDoor && state.doorwayOpen ? 0.83 : 0.72;
        }
        ctx.globalAlpha = alpha;
        ctx.fillStyle = fill;
        roundedRect(x + border, y + border, cell - border * 2, cell - border * 2, Math.max(1, cell * 0.08));
        ctx.fill();
        ctx.globalAlpha = 1;

        if (isHiddenWall && !(isDoor && state.doorwayOpen)) {
          ctx.save();
          ctx.strokeStyle = "rgba(7, 17, 31, 0.28)";
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(x + cell * 0.2, y + cell * 0.85);
          ctx.lineTo(x + cell * 0.85, y + cell * 0.2);
          ctx.stroke();
          ctx.restore();
        }
      }
    }

    const sy = oy + 5.5 * cell;
    const sx = ox + 2.5 * cell;
    const gx = ox + 12.5 * cell;
    const gy = sy;
    const sampleValid = state.doorwayOpen && state.radius <= 1;
    const radiusPx = 5 + state.radius * Math.max(2, cell * 0.15);

    if (sampleValid) {
      ctx.save();
      ctx.strokeStyle = palette.path;
      ctx.lineWidth = Math.max(2, cell * 0.11);
      ctx.lineCap = "round";
      ctx.setLineDash([Math.max(4, cell * 0.28), Math.max(3, cell * 0.2)]);
      ctx.lineDashOffset = -(time * 0.025) % 20;
      ctx.beginPath();
      ctx.moveTo(sx, sy);
      ctx.lineTo(gx, gy);
      ctx.stroke();
      ctx.restore();

      const progress = (time * 0.00018) % 1;
      const px = sx + (gx - sx) * progress;
      ctx.fillStyle = palette.path;
      ctx.shadowColor = palette.path;
      ctx.shadowBlur = 14;
      ctx.beginPath();
      ctx.arc(px, sy, Math.max(2.5, cell * 0.12), 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;
    } else {
      ctx.save();
      ctx.strokeStyle = palette.rose;
      ctx.globalAlpha = 0.55;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 5]);
      ctx.beginPath();
      ctx.moveTo(sx, sy);
      ctx.lineTo(gx, gy);
      ctx.stroke();
      ctx.restore();
    }

    for (const [x, label, color] of [[sx, "s", palette.freeLight], [gx, "g", palette.path]]) {
      ctx.fillStyle = color;
      ctx.shadowColor = color;
      ctx.shadowBlur = 11;
      ctx.beginPath();
      ctx.arc(x, sy, radiusPx, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;
      drawLabel(label, x, sy + 0.5, palette.background, Math.max(8, cell * 0.28), "center");
    }
  }

  function drawGraph(panel) {
    const values = eventProbabilities();
    const inset = { left: 42, right: 18, top: 55, bottom: 45 };
    const x0 = panel.x + inset.left;
    const y0 = panel.y + panel.h - inset.bottom;
    const graphW = panel.w - inset.left - inset.right;
    const graphH = panel.h - inset.top - inset.bottom;
    drawLabel("EVENT RELIABILITY  q(s,g,r|X)", panel.x + 14, panel.y + 18, palette.freeLight, 9);
    drawLabel(`context prior = ${state.context === 0 ? "0.20" : "0.80"}`, panel.x + panel.w - 14, panel.y + 18, palette.muted, 9, "right");

    ctx.strokeStyle = palette.gridStrong;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x0, panel.y + inset.top);
    ctx.lineTo(x0, y0);
    ctx.lineTo(x0 + graphW, y0);
    ctx.stroke();

    for (let tick = 0; tick <= 4; tick += 1) {
      const value = tick / 4;
      const y = y0 - value * graphH;
      ctx.strokeStyle = palette.grid;
      ctx.beginPath();
      ctx.moveTo(x0, y);
      ctx.lineTo(x0 + graphW, y);
      ctx.stroke();
      drawLabel(value.toFixed(2), x0 - 7, y, palette.muted, 8, "right");
    }

    const slot = graphW / values.length;
    values.forEach((value, index) => {
      const barW = Math.min(38, slot * 0.46);
      const x = x0 + slot * (index + 0.5) - barW / 2;
      const barH = value * graphH;
      const selected = index === state.radius;
      const gradient = ctx.createLinearGradient(0, y0 - barH, 0, y0);
      gradient.addColorStop(0, selected ? palette.path : palette.freeLight);
      gradient.addColorStop(1, selected ? "#5f8a45" : palette.free);
      ctx.globalAlpha = selected ? 1 : 0.42;
      ctx.fillStyle = gradient;
      roundedRect(x, y0 - barH, barW, Math.max(2, barH), 4);
      ctx.fill();
      ctx.globalAlpha = 1;
      drawLabel(`r=${index}`, x + barW / 2, y0 + 19, selected ? palette.text : palette.muted, 9, "center");
      drawLabel(value.toFixed(2), x + barW / 2, y0 - barH - 11, selected ? palette.path : palette.muted, 9, "center");
    });

    drawLabel("radius increases → support shrinks", x0 + graphW / 2, panel.y + panel.h - 13, palette.muted, 8, "center");
  }

  function drawFrame(time) {
    if (state.playing && time - state.lastWorldSwitch > 2800) {
      state.doorwayOpen = !state.doorwayOpen;
      state.lastWorldSwitch = time;
      updateReadout();
    }

    const gradient = ctx.createLinearGradient(0, 0, width, height);
    gradient.addColorStop(0, palette.background);
    gradient.addColorStop(1, "#07121f");
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, width, height);

    const margin = width < 540 ? 18 : 26;
    const gap = width < 540 ? 12 : 16;
    const stacked = width < 620;
    let mapPanel;
    let graphPanel;
    if (stacked) {
      const available = height - margin * 2 - gap;
      mapPanel = { x: margin, y: margin + 14, w: width - margin * 2, h: available * 0.57 };
      graphPanel = { x: margin, y: mapPanel.y + mapPanel.h + gap, w: width - margin * 2, h: available * 0.43 - 14 };
    } else {
      const contentW = width - margin * 2 - gap;
      mapPanel = { x: margin, y: margin + 12, w: contentW * 0.61, h: height - margin * 2 - 12 };
      graphPanel = { x: mapPanel.x + mapPanel.w + gap, y: mapPanel.y, w: contentW * 0.39, h: mapPanel.h };
    }

    for (const panel of [mapPanel, graphPanel]) {
      ctx.fillStyle = palette.panel;
      ctx.strokeStyle = palette.gridStrong;
      ctx.lineWidth = 1;
      roundedRect(panel.x, panel.y, panel.w, panel.h, 12);
      ctx.fill();
      ctx.stroke();
    }
    drawMap(mapPanel, time);
    drawGraph(graphPanel);
    requestAnimationFrame(drawFrame);
  }

  const resizeObserver = new ResizeObserver(resizeCanvas);
  resizeObserver.observe(canvas);
  resizeCanvas();
  updateReadout();
  setPlaying(state.playing);
  requestAnimationFrame(drawFrame);

  const fallbackSnapshot = {
    report_mtime_utc: null,
    protocol: { seed: 20260827, radii_cells: [0, 1, 2], posterior_samples: 64 },
    event_metrics: [
      { id: "independent_bernoulli", name: "Independent cell completion", brier: 0.183172, nll: 1.031629, ece: 0.176242, false_safe_rate_at_08: 0.25 },
      { id: "direct_query_mlp", name: "Direct query predictor", brier: 0.169888, nll: 0.522382, ece: 0.089128, false_safe_rate_at_08: null },
      { id: "PathRel_correlated_event", name: "ConPath correlated event (oracle proxy)", brier: 0.102373, nll: 0.324969, ece: 0.032484, false_safe_rate_at_08: 0.14741 },
    ],
    headline: {
      oracle_proxy: { brier: 0.102373, ece: 0.032484, count: 1152 },
      independent_cells: { brier: 0.183172 },
      direct_query: { brier: 0.169888 },
    },
    neural_latest: null,
  };

  function format(value, digits = 4) {
    return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "—";
  }

  function renderMetrics(data) {
    const oracle = data.headline?.oracle_proxy || fallbackSnapshot.headline.oracle_proxy;
    const independent = data.headline?.independent_cells || fallbackSnapshot.headline.independent_cells;
    const direct = data.headline?.direct_query || fallbackSnapshot.headline.direct_query;
    const improvementIndependent = 100 * (1 - oracle.brier / independent.brier);
    const improvementDirect = 100 * (1 - oracle.brier / direct.brier);
    $("#metric-cards").innerHTML = `
      <article class="metric-card featured"><span class="metric-label">Oracle proxy · Brier ↓</span><strong class="metric-value">${format(oracle.brier)}</strong><span class="metric-detail">held-out event samples <span class="metric-delta">n = ${oracle.count || 1152}</span></span></article>
      <article class="metric-card"><span class="metric-label">vs independent cells</span><strong class="metric-value">${format(improvementIndependent, 1)}%</strong><span class="metric-detail">relative Brier reduction <span class="metric-delta">${format(independent.brier)} → ${format(oracle.brier)}</span></span></article>
      <article class="metric-card"><span class="metric-label">vs direct q predictor</span><strong class="metric-value">${format(improvementDirect, 1)}%</strong><span class="metric-detail">relative Brier reduction <span class="metric-delta">ECE ${format(oracle.ece)}</span></span></article>`;

    const rows = data.event_metrics || fallbackSnapshot.event_metrics;
    $("#row-count").textContent = `${rows.length} audited methods`;
    $("#metrics-body").innerHTML = rows.map((row) => {
      const featured = row.id === "PathRel_correlated_event";
      return `<tr class="${featured ? "featured-row" : ""}"><td>${row.name}${featured ? " · proxy" : ""}</td><td>${format(row.brier)}</td><td>${format(row.nll)}</td><td>${format(row.ece)}</td><td>${format(row.false_safe_rate_at_08)}</td></tr>`;
    }).join("");

    const neural = data.neural_latest;
    if (!neural) {
      $("#neural-summary").innerHTML = `<p>No compact neural report was found when this snapshot was built. The public claim remains blocked.</p><p class="source-line">Run scripts/train_p0_neural.py, then rebuild the site snapshot.</p>`;
    } else {
      const event = neural.event || {};
      const paperLabel = neural.paper_result ? "paper result" : "not a paper result";
      $("#neural-summary").innerHTML = `
        <p>The latest CUDA snapshot is an active failure-analysis run: <strong>${paperLabel}</strong>.</p>
        <div class="neural-stats">
          <div class="neural-stat"><span>Brier ↓</span><strong>${format(event.brier)}</strong></div>
          <div class="neural-stat"><span>ECE ↓</span><strong>${format(event.ece)}</strong></div>
          <div class="neural-stat"><span>NLL ↓</span><strong>${format(event.nll)}</strong></div>
          <div class="neural-stat"><span>false-safe</span><strong>${format(event["false_safe_rate@0.8"])}</strong></div>
        </div>
        <p class="source-line">source: ${neural.source}</p>`;
    }

    const protocol = data.protocol || {};
    const date = data.report_mtime_utc ? new Date(data.report_mtime_utc).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }) : "bundled fallback";
    const templates = Array.isArray(protocol.test_templates) ? protocol.test_templates.length : protocol.test_templates;
    $("#generated-line").textContent = `Tracked snapshot · seed ${protocol.seed ?? "—"} · ${templates ?? "—"} held-out templates · ${protocol.posterior_samples ?? "—"} posterior samples · ${date}`;
  }

  async function loadSnapshot() {
    if (window.CONPATH_SNAPSHOT) return window.CONPATH_SNAPSHOT;
    try {
      const response = await fetch("data/p0_metrics.json", { cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return await response.json();
    } catch (error) {
      console.warn("Using bundled metric fallback:", error);
      return fallbackSnapshot;
    }
  }

  loadSnapshot().then(renderMetrics);
})();
