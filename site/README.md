# ConPath demo site

This directory is a static, GitHub Pages-ready research demo. It is intentionally additive: the
Python package and experiment protocol remain under `src/`, `scripts/`, and `tests/`.

## Refresh the tracked snapshot

From the repository root, after running the P0 evaluator:

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_p0.py --output-dir results/p0_death_test
PYTHONPATH=src .venv/bin/python scripts/build_demo_site.py
```

The builder copies the reproducible `comparison.svg`, `reliability.svg`, and `metrics.csv` outputs
into `site/`, and writes a compact `data/p0_metrics.json` plus a browser-local JS mirror. The JS
mirror means `index.html` also works when opened directly from disk.

## Refresh the video

The checked-in video is generated from the same synthetic contract. On this workstation, Pillow is
available in the system interpreter and FFmpeg is supplied by the local Node tool cache:

```bash
/usr/bin/python3 scripts/build_demo_video.py
```

Use `--ffmpeg /path/to/ffmpeg` if FFmpeg is installed elsewhere. The script writes
`site/assets/conpath_p0_demo.mp4` and its poster image; no experiment output is modified.

## Preview locally

```bash
python3 -m http.server 8000 --directory site
```

Open <http://127.0.0.1:8000>. A local HTTP server is recommended because it also exercises the same
asset paths used by GitHub Pages.

## GitHub Pages

`.github/workflows/deploy-pages.yml` publishes this directory on every push to `main`. Enable Pages
once in the repository settings and choose **GitHub Actions** as the source. The workflow only
uploads tracked files; it does not rerun GPU experiments on the hosted runner.
