(() => {
  "use strict";

  const number = (value, digits = 3) => Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "—";
  const integer = (value) => Number.isFinite(Number(value)) ? Number(value).toLocaleString("en-US") : "—";
  const percent = (value, digits = 1) => Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(digits)}%` : "—";
  const escapeHtml = (value) => String(value).replace(/[&<>'"]/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  })[character]);

  const snapshot = window.CONPATH_REAL_PILOT;
  const body = document.querySelector("#metrics-body");
  const headline = document.querySelector("#best-brier");
  const summary = document.querySelector("#result-summary");
  if (!snapshot || !body) {
    if (body) body.innerHTML = '<tr><td colspan="5">Real-data snapshot not found. Rebuild with <code>--publish-site</code>.</td></tr>';
  } else {
    const metrics = Array.isArray(snapshot.metrics) ? snapshot.metrics : [];
    body.innerHTML = metrics.map((row) => {
      const highlight = row.id === "correlated_temporal" ? " class=\"highlight\"" : "";
      return `<tr${highlight}><td>${escapeHtml(row.name)}</td><td>${number(row.brier)}</td><td>${number(row.nll)}</td><td>${number(row.ece)}</td><td>${number(row["false_safe_rate@0.8"])}</td></tr>`;
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
  }

  const audit = window.CONPATH_FLATLANDS_AUDIT;
  const strataBody = document.querySelector("#flatlands-strata-body");
  const overlapBody = document.querySelector("#flatlands-overlap-body");
  if (!audit || !strataBody) {
    if (strataBody) strataBody.innerHTML = '<tr><td colspan="7">FlatLands audit snapshot not found. Rebuild the project site.</td></tr>';
    if (overlapBody) overlapBody.innerHTML = '<tr><td colspan="3">FlatLands provenance audit not found.</td></tr>';
  } else {
    const reachability = (row, radius) => percent(((row.reachable_by_radius_cells || {})[String(radius)] || {}).rate);
    const radii = Array.isArray(audit.radii_cells) ? audit.radii_cells : [0, 10, 20];
    strataBody.innerHTML = (audit.strata || []).map((row) => (
      `<tr><td>${escapeHtml(row.split)} · ${escapeHtml(row.source)}</td>`
      + `<td>${integer(row.retained_queries)}</td>`
      + `<td>${reachability(row, radii[0])}</td>`
      + `<td>${reachability(row, radii[1])}</td>`
      + `<td>${reachability(row, radii[2])}</td>`
      + `<td>${percent(row.scene_weighted_failure_rate)}</td>`
      + `<td><span class="${row.gate_passed ? "gate-pass" : "gate-hold"}">${row.gate_passed ? "PASS" : "HOLD"}</span></td></tr>`
    )).join("") || '<tr><td colspan="7">No gated source strata in snapshot.</td></tr>';

    if (overlapBody) {
      overlapBody.innerHTML = (audit.official_split_overlap || []).map((row) => (
        `<tr><td>${escapeHtml(row.pair)}</td><td class="overlap-bad">${integer(row.official_scene_overlap)}</td><td class="overlap-good">${integer(row.provenance_scene_overlap)}</td></tr>`
      )).join("") || '<tr><td colspan="3">No split-pair audit rows.</td></tr>';
    }

    const gates = audit.gated_strata || {};
    const observations = document.querySelector("#flatlands-observations");
    const queries = document.querySelector("#flatlands-queries");
    const strata = document.querySelector("#flatlands-strata");
    const gate = document.querySelector("#flatlands-gate");
    const claim = document.querySelector("#flatlands-claim");
    if (observations) observations.textContent = `${integer(audit.selected_observations)} scenes`;
    if (queries) queries.textContent = integer((audit.totals || {}).retained_valid_endpoint_queries);
    if (strata) strata.textContent = `${integer(gates.passed)} / ${integer(gates.total)} pass`;
    if (gate) gate.textContent = `${integer(gates.passed)}/${integer(gates.total)}`;
    if (claim) claim.textContent = audit.claim_boundary || "Bounded data audit; not a trained-model result.";
  }

  const baselines = window.CONPATH_FLATLANDS_BASELINES;
  const baselineBody = document.querySelector("#flatlands-baseline-body");
  if (!baselines || !baselineBody) {
    if (baselineBody) baselineBody.innerHTML = '<tr><td colspan="6">Validation baseline snapshot not found. Rebuild the project site.</td></tr>';
  } else {
    const methods = Array.isArray(baselines.methods) ? baselines.methods : [];
    const overall = (method) => ((method.overall || {}).scene_weighted || {});
    const best = methods.slice().sort((a, b) => Number(overall(a).brier) - Number(overall(b).brier))[0];
    baselineBody.innerHTML = methods.map((method) => {
      const metric = overall(method);
      const highlight = best && method.id === best.id ? " class=\"highlight\"" : "";
      return `<tr${highlight}><td>${escapeHtml(method.name)}</td><td>${number(metric.brier)}</td><td>${number(metric.nll)}</td><td>${number(metric.ece)}</td><td>${percent(metric["false_safe_rate@0.8"])}</td><td>${percent(metric["high_confidence_safe_coverage@0.8"])}</td></tr>`;
    }).join("") || '<tr><td colspan="6">No validation baseline methods in snapshot.</td></tr>';

    const independent = methods.find((method) => method.id === "independent_cell_completion_k32");
    const methodsCard = document.querySelector("#flatlands-baseline-methods");
    const bestCard = document.querySelector("#flatlands-baseline-best");
    const independentCard = document.querySelector("#flatlands-baseline-independent");
    const testCard = document.querySelector("#flatlands-baseline-test");
    const headlineCard = document.querySelector("#flatlands-baseline-headline");
    const claimCard = document.querySelector("#flatlands-baseline-claim");
    if (methodsCard) methodsCard.textContent = integer(methods.length);
    if (bestCard) bestCard.textContent = best ? number(overall(best).brier, 4) : "—";
    if (independentCard) independentCard.textContent = independent ? number(overall(independent).brier, 4) : "—";
    if (testCard) testCard.textContent = baselines.test_evaluated ? "EVALUATED" : "LOCKED";
    if (headlineCard) headlineCard.textContent = best ? number(overall(best).brier, 4) : "—";
    if (claimCard) claimCard.textContent = baselines.claim_boundary || "Validation-only baseline diagnostics; no test result is published.";
  }
})();
