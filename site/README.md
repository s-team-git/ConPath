# ConPath project page

This directory is a static, GitHub Pages-ready academic project page. The primary media are derived
from the real TUM RGB-D Freiburg1/desk sequence: the RGB/depth/pose composite, the pilot video, and
the event-metric figures. It also publishes the bounded FlatLands data-gate snapshot and figures,
explicitly labelled as target statistics rather than model results. No synthetic video or synthetic
figure is referenced by `index.html`.

## Refresh the real-data page

From the repository root, after downloading and extracting the TUM sequence:

```bash
/usr/bin/python3 scripts/run_tum_rgbd_pilot.py --publish-site
/usr/bin/python3 scripts/build_demo_site.py
# Optional torch-only integration smoke on the same real BEV hand-off:
PYTHONPATH=src .venv/bin/python scripts/run_tum_rgbd_model_smoke.py --device cpu
```

The first command writes the reproducible pilot report under
`results/tum_rgbd_freiburg1_desk_pilot/` and copies compact derived assets into `site/`. The second
command rebuilds `data/tum_rgbd_pilot.json`, `data/flatlands_audit.json`, their browser-local JS
mirrors, and the two FlatLands audit SVGs. It expects the bounded query and provenance audit reports
under `results/`; use `--skip-flatlands` only when intentionally rebuilding the TUM-only page. Raw
datasets remain under the ignored `data/raw/` directory and are never committed.

The legacy synthetic P0 snapshot can still be regenerated for development with
`scripts/build_demo_site.py --include-legacy`, but it is intentionally excluded from the public page.

## Preview locally

```bash
python3 -m http.server 8000 --directory site
```

Open <http://127.0.0.1:8000>. A local HTTP server exercises the same relative asset paths used by
GitHub Pages.

## GitHub Pages

`.github/workflows/deploy-pages.yml` publishes this directory on every push to `main`. Enable Pages
once in repository settings and choose **GitHub Actions** as the source. The workflow uploads tracked
files; it does not download datasets or run GPU experiments on the hosted runner.

## Attribution and claim boundary

The TUM RGB-D sequence is credited in the page and linked to the official source. Its RGB/depth and
motion-capture trajectory support a geometric reference-map pilot, not traversability or collision
ground truth. The FlatLands section uses only the scene-disjoint upstream provenance split because
the physical archive split leaks scenes, and remains a data audit until trained baselines exist. See
[`REAL_DATA_PILOT.md`](../REAL_DATA_PILOT.md) and [`P1_DATA_AUDIT.md`](../P1_DATA_AUDIT.md) for exact
protocols and limitations.
