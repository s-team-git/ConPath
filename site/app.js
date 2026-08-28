(() => {
  "use strict";

  const snapshot = window.CONPATH_REAL_PILOT;
  const body = document.querySelector("#metrics-body");
  const headline = document.querySelector("#best-brier");
  const summary = document.querySelector("#result-summary");
  if (!snapshot || !body) {
    if (body) body.innerHTML = '<tr><td colspan="5">Real-data snapshot not found. Rebuild with <code>--publish-site</code>.</td></tr>';
    return;
  }

  const number = (value, digits = 3) => Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "—";
  const metrics = Array.isArray(snapshot.metrics) ? snapshot.metrics : [];
  body.innerHTML = metrics.map((row) => {
    const highlight = row.id === "correlated_temporal" ? " class=\"highlight\"" : "";
    return `<tr${highlight}><td>${row.name}</td><td>${number(row.brier)}</td><td>${number(row.nll)}</td><td>${number(row.ece)}</td><td>${number(row["false_safe_rate@0.8"])}</td></tr>`;
  }).join("") || '<tr><td colspan="5">No metric rows in the tracked snapshot.</td></tr>';

  const correlated = metrics.find((row) => row.id === "correlated_temporal");
  if (correlated && headline) headline.textContent = number(correlated.brier, 4);
  if (summary) {
    const dataset = snapshot.dataset || {};
    const protocol = snapshot.protocol || {};
    const split = protocol.temporal_split || {};
    summary.textContent = `${dataset.name || "TUM RGB-D"}: ${dataset.sampled_frames || "—"} sampled frames, ${split.future_query_frames || "—"} future frames, ${protocol.queries || "—"} query pairs, and radii ${(protocol.radii_cells || []).join(", ")}.`;
  }

  // Keep the metadata cards synchronized with the generated report without inventing author or
  // dataset claims when a future pilot changes the sequence.
  const cards = document.querySelectorAll("#pilot-meta .meta-card");
  if (cards.length >= 3) {
    const dataset = snapshot.dataset || {};
    const protocol = snapshot.protocol || {};
    const map = snapshot.map || {};
    cards[0].querySelector("strong").textContent = dataset.name || "TUM RGB-D";
    cards[0].querySelector("small").textContent = `${dataset.rgb_frames_available || "—"} RGB / ${dataset.depth_frames_available || "—"} depth frames`;
    cards[1].querySelector("strong").textContent = `${dataset.sampled_frames || "—"} sampled frames`;
    cards[1].querySelector("small").textContent = `${(protocol.temporal_split || {}).observed_prefix_frames || "—"} prefix / ${(protocol.temporal_split || {}).future_query_frames || "—"} future`;
    cards[2].querySelector("strong").textContent = `${number((protocol.raster || {}).resolution_m, 2)} m cells`;
    cards[2].querySelector("small").textContent = `${map.height_cells || "—"} × ${map.width_cells || "—"} BEV raster`;
  }
})();
