#!/usr/bin/env python3
"""Render the ConPath synthetic P0 walkthrough video.

The renderer uses Pillow for a small, deterministic raster animation and pipes PNG frames directly
to FFmpeg.  It never reads or writes checkpoints.  The resulting MP4 is a communication artifact:
the final frame repeats the tracked report values and labels the correlated row as an *oracle
proxy*, keeping the demo useful without overstating the neural experiment.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from io import BytesIO
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = PROJECT_ROOT / "results" / "p0_death_test" / "report.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "site" / "assets" / "conpath_p0_demo.mp4"
DEFAULT_POSTER = PROJECT_ROOT / "site" / "assets" / "conpath_p0_demo_poster.png"

COLORS = {
    "bg0": (7, 17, 31),
    "bg1": (8, 28, 43),
    "panel": (12, 29, 47),
    "panel2": (15, 38, 58),
    "line": (57, 94, 117),
    "grid": (36, 71, 91),
    "text": (241, 247, 251),
    "muted": (143, 164, 183),
    "subtle": (96, 119, 139),
    "cyan": (105, 228, 232),
    "cyan_dark": (47, 143, 164),
    "blocked": (23, 41, 59),
    "lime": (185, 239, 115),
    "amber": (255, 200, 117),
    "rose": (255, 135, 150),
    "ink": (5, 13, 23),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--poster", type=Path, default=DEFAULT_POSTER)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--ffmpeg", type=Path, default=None, help="optional FFmpeg executable")
    return parser.parse_args()


def require_pillow() -> tuple[Any, Any, Any]:
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on the host interpreter
        raise SystemExit(
            "Pillow is required for the demo video. On the development workstation run "
            "the script with /usr/bin/python3, or install Pillow in the active interpreter."
        ) from exc
    return Image, ImageDraw, ImageFont


def find_ffmpeg(explicit: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    resolved = shutil.which("ffmpeg")
    if resolved:
        candidates.append(Path(resolved))
    # The workstation's Node tool cache includes a static FFmpeg binary.  Keep this discovery
    # optional so the script remains portable to a normal system installation.
    candidates.extend(
        Path("/home/hairo/.local/share/pnpm/store").glob(
            "**/node_modules/@ffmpeg-installer/linux-x64/ffmpeg"
        )
    )
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_mode & 0o111:
            return candidate
    raise SystemExit(
        "FFmpeg was not found. Install ffmpeg or pass --ffmpeg /path/to/ffmpeg."
    )


def read_report(path: Path) -> dict[str, float]:
    fallback = {
        "oracle_brier": 0.1023726993,
        "independent_brier": 0.1831715902,
        "direct_brier": 0.1698875835,
        "oracle_ece": 0.0324842665,
    }
    if not path.exists():
        return fallback
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        metrics = report.get("event_metrics", {})
        values = {
            "oracle_brier": metrics.get("PathRel_correlated_event", {}).get("brier"),
            "independent_brier": metrics.get("independent_bernoulli", {}).get("brier"),
            "direct_brier": metrics.get("direct_query_mlp", {}).get("brier"),
            "oracle_ece": metrics.get("PathRel_correlated_event", {}).get("ece"),
        }
        if all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values.values()):
            return {key: float(value) for key, value in values.items()}
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    return fallback


def font(ImageFont: Any, size: int, bold: bool = False) -> Any:
    names = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for name in names if bold else names[::-1]:
        path = Path(name)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def mix(color: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    return tuple(max(0, min(255, int(channel * amount))) for channel in color)


def draw_text(draw: Any, xy: tuple[float, float], text: str, fnt: Any, fill: tuple[int, int, int], anchor: str = "la") -> None:
    draw.text(xy, text, font=fnt, fill=fill, anchor=anchor)


def rounded(draw: Any, box: tuple[float, float, float, float], radius: int, fill: Any, outline: Any = None, width: int = 1) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def dashed_line(draw: Any, start: tuple[float, float], end: tuple[float, float], fill: Any, width: int, dash: int = 9, gap: int = 7, offset: float = 0.0) -> None:
    sx, sy = start
    ex, ey = end
    dx, dy = ex - sx, ey - sy
    length = max(1.0, math.hypot(dx, dy))
    ux, uy = dx / length, dy / length
    cursor = -float(offset) % (dash + gap)
    while cursor < length:
        a = max(0.0, cursor)
        b = min(length, cursor + dash)
        if b > a:
            draw.line((sx + ux * a, sy + uy * a, sx + ux * b, sy + uy * b), fill=fill, width=width)
        cursor += dash + gap


def make_background(Image: Any, ImageDraw: Any, width: int, height: int) -> Any:
    image = Image.new("RGB", (width, height))
    pixels = image.load()
    for y in range(height):
        fy = y / max(1, height - 1)
        for x in range(width):
            fx = x / max(1, width - 1)
            glow = max(0.0, 1.0 - math.hypot(fx - 0.78, fy - 0.08) * 1.25)
            pixels[x, y] = tuple(
                int(COLORS["bg0"][i] * (1 - 0.25 * fy) + COLORS["bg1"][i] * 0.25 * fy + (8 + 15 * glow) * (i == 1))
                for i in range(3)
            )
    draw = ImageDraw.Draw(image, "RGBA")
    for x in range(0, width, 48):
        draw.line((x, 0, x, height), fill=(120, 172, 196, 13), width=1)
    for y in range(0, height, 48):
        draw.line((0, y, width, y), fill=(120, 172, 196, 13), width=1)
    return image


def render_frame(
    Image: Any,
    ImageDraw: Any,
    ImageFont: Any,
    width: int,
    height: int,
    t: float,
    metrics: dict[str, float],
    background: Any,
) -> Any:
    # The gradient/grid is static; copying one precomputed image is substantially faster than
    # recomputing half a million pixels for every animation frame.
    image = background.copy()
    draw = ImageDraw.Draw(image, "RGBA")
    regular = font(ImageFont, max(12, width // 76))
    small = font(ImageFont, max(9, width // 108))
    tiny = font(ImageFont, max(8, width // 132), bold=True)
    heading = font(ImageFont, max(22, width // 31), bold=True)
    subheading = font(ImageFont, max(13, width // 67), bold=True)

    margin = int(width * 0.048)
    left, top, right = margin, int(height * 0.19), width - margin
    panel_gap = int(width * 0.025)
    map_right = int(width * 0.58)
    map_box = (left, top, map_right, int(height * 0.86))
    graph_box = (map_right + panel_gap, top, right, int(height * 0.86))

    # Header and stable identity.
    draw_text(draw, (left, int(height * 0.07)), "CONPATH", tiny, COLORS["cyan"])
    draw_text(draw, (left, int(height * 0.105)), "Connectivity-calibrated path reliability", heading, COLORS["text"])
    draw_text(draw, (right, int(height * 0.079)), "SYNTHETIC P0 AUDIT", tiny, COLORS["lime"], anchor="ra")
    draw_text(draw, (right, int(height * 0.112)), "oracle proxy · template held out", small, COLORS["muted"], anchor="ra")

    for box in (map_box, graph_box):
        rounded(draw, box, 14, fill=COLORS["panel"], outline=(*COLORS["line"], 170), width=1)

    phase = (t % 8.0) / 8.0
    context = 0 if phase < 0.5 else 1
    prior = 0.2 if context == 0 else 0.8
    local_t = t % 2.0
    radius = min(2, int(local_t / 0.66))
    doorway_open = int(t * 0.85) % 2 == 0
    sample_valid = doorway_open and radius <= 1
    probabilities = [prior, prior * 0.72, prior * 0.34]

    # Map panel.
    mx0, my0, mx1, my1 = map_box
    draw_text(draw, (mx0 + 18, my0 + 22), "FIXED OBSERVATION  X", tiny, COLORS["cyan"])
    draw_text(draw, (mx1 - 18, my0 + 22), "HIDDEN WORLD SAMPLE", tiny, COLORS["muted"], anchor="ra")
    cols, rows = 15, 9
    cell = min((mx1 - mx0 - 45) / cols, (my1 - my0 - 86) / rows)
    ox = mx0 + (mx1 - mx0 - cell * cols) / 2
    oy = my0 + 53
    wall_col = 7
    door_rows = {3, 4, 5}
    for row in range(rows):
        for col in range(cols):
            x, y = ox + col * cell, oy + row * cell
            outer = row in {0, rows - 1}
            wall = col == wall_col and not outer
            door = wall and row in door_rows
            if outer:
                fill = COLORS["blocked"]
                alpha = 245
            elif wall:
                fill = COLORS["cyan"] if door and doorway_open else COLORS["amber"]
                alpha = 218 if door and doorway_open else 180
            else:
                fill = COLORS["cyan_dark"]
                alpha = 130
            rounded(draw, (x + 1, y + 1, x + cell - 1, y + cell - 1), max(2, int(cell * 0.09)), fill=(*fill, alpha))
            if wall and not (door and doorway_open):
                draw.line((x + cell * 0.22, y + cell * 0.82, x + cell * 0.82, y + cell * 0.22), fill=(*COLORS["ink"], 75), width=1)

    sy = oy + 4.5 * cell
    sx = ox + 2.5 * cell
    gx = ox + 12.5 * cell
    pulse = 1.0 + 0.12 * math.sin(t * 4.2)
    if sample_valid:
        dashed_line(draw, (sx, sy), (gx, sy), COLORS["lime"], max(2, int(cell * 0.11)), dash=max(7, int(cell * 0.3)), gap=max(5, int(cell * 0.2)), offset=t * 35)
        progress = (t * 0.22) % 1.0
        px = sx + (gx - sx) * progress
        draw.ellipse((px - 4 * pulse, sy - 4 * pulse, px + 4 * pulse, sy + 4 * pulse), fill=(*COLORS["lime"], 245))
    else:
        dashed_line(draw, (sx, sy), (gx, sy), COLORS["rose"], 2, dash=5, gap=7, offset=t * 16)

    for x, label, color in ((sx, "s", COLORS["cyan"]), (gx, "g", COLORS["lime"])):
        radius_px = max(7, int(cell * 0.2))
        draw.ellipse((x - radius_px, sy - radius_px, x + radius_px, sy + radius_px), fill=color)
        draw_text(draw, (x, sy + 1), label, tiny, COLORS["ink"], anchor="mm")
    door_text = "doorway OPEN" if doorway_open else "doorway BLOCKED"
    door_color = COLORS["lime"] if doorway_open else COLORS["rose"]
    draw_text(draw, (mx0 + 18, my1 - 23), door_text, small, door_color)
    draw_text(draw, (mx1 - 18, my1 - 23), f"context {context} · prior {prior:.2f}", small, COLORS["muted"], anchor="ra")

    # Graph panel.
    gx0, gy0, gx1, gy1 = graph_box
    draw_text(draw, (gx0 + 18, gy0 + 22), "EVENT RELIABILITY", tiny, COLORS["cyan"])
    draw_text(draw, (gx1 - 18, gy0 + 22), "q(s,g,r | X)", tiny, COLORS["muted"], anchor="ra")
    chart_l, chart_r = gx0 + 52, gx1 - 21
    chart_t, chart_b = gy0 + 65, gy1 - 55
    for tick in range(5):
        value = tick / 4
        y = chart_b - value * (chart_b - chart_t)
        draw.line((chart_l, y, chart_r, y), fill=(*COLORS["grid"], 165), width=1)
        draw_text(draw, (chart_l - 8, y), f"{value:.2f}", tiny, COLORS["subtle"], anchor="ra")
    draw.line((chart_l, chart_t, chart_l, chart_b), fill=(*COLORS["line"], 200), width=1)
    draw.line((chart_l, chart_b, chart_r, chart_b), fill=(*COLORS["line"], 200), width=1)
    slot = (chart_r - chart_l) / 3
    for index, value in enumerate(probabilities):
        bar_w = min(34, slot * 0.46)
        x = chart_l + slot * (index + 0.5) - bar_w / 2
        y = chart_b - value * (chart_b - chart_t)
        selected = index == radius
        fill = COLORS["lime"] if selected else COLORS["cyan"]
        alpha = 245 if selected else 105
        rounded(draw, (x, y, x + bar_w, chart_b), 5, fill=(*fill, alpha))
        draw_text(draw, (x + bar_w / 2, chart_b + 18), f"r={index}", tiny, COLORS["text"] if selected else COLORS["muted"], anchor="mm")
        draw_text(draw, (x + bar_w / 2, y - 11), f"{value:.2f}", tiny, fill, anchor="mm")
    draw_text(draw, ((chart_l + chart_r) / 2, gy1 - 22), "larger footprint → fewer support-valid worlds", tiny, COLORS["subtle"], anchor="mm")

    # A compact metric callout fades in during the final quarter of the loop.
    if phase >= 0.72:
        fade = min(1.0, (phase - 0.72) / 0.12)
        alpha = int(235 * fade)
        callout = (left + 18, int(height * 0.885), right - 18, int(height * 0.965))
        rounded(draw, callout, 9, fill=(*COLORS["panel2"], alpha), outline=(*COLORS["lime"], int(120 * fade)), width=1)
        draw_text(draw, (callout[0] + 13, callout[1] + 17), "P0 BRIER ↓", tiny, (*COLORS["muted"], alpha))
        draw_text(draw, (callout[0] + 106, callout[1] + 17), f"oracle proxy {metrics['oracle_brier']:.3f}", small, (*COLORS["lime"], alpha))
        draw_text(draw, (callout[0] + 300, callout[1] + 17), f"independent {metrics['independent_brier']:.3f}", small, (*COLORS["cyan"], alpha))
        draw_text(draw, (callout[2] - 13, callout[1] + 17), "proxy ≠ trained neural result", tiny, (*COLORS["amber"], alpha), anchor="ra")

    draw_text(draw, (left, height - 14), "ConPath research preview · same visible observation, correlated hidden topology", tiny, COLORS["subtle"])
    draw_text(draw, (right, height - 14), "conpath / P0", tiny, COLORS["subtle"], anchor="ra")
    return image


def encode(Image: Any, ImageDraw: Any, ImageFont: Any, args: argparse.Namespace, metrics: dict[str, float], ffmpeg: Path) -> None:
    if args.fps <= 0 or args.duration <= 0:
        raise SystemExit("--fps and --duration must be positive")
    if args.width < 320 or args.height < 180:
        raise SystemExit("video dimensions are too small for the demo layout")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.poster.parent.mkdir(parents=True, exist_ok=True)

    command = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "-r",
        str(args.fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "25",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-metadata",
        "title=ConPath synthetic P0 audit walkthrough",
        str(args.output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    frame_count = max(1, int(round(args.duration * args.fps)))
    background = make_background(Image, ImageDraw, args.width, args.height)
    try:
        for frame_index in range(frame_count):
            t = frame_index / args.fps
            frame = render_frame(Image, ImageDraw, ImageFont, args.width, args.height, t, metrics, background)
            if frame_index == 0:
                frame.save(args.poster, format="PNG", optimize=True)
            buffer = BytesIO()
            frame.save(buffer, format="PNG", optimize=False)
            process.stdin.write(buffer.getvalue())
    except BrokenPipeError as exc:
        process.stdin.close()
        process.wait()
        raise SystemExit("FFmpeg stopped while receiving frames") from exc
    finally:
        try:
            process.stdin.close()
        except BrokenPipeError:
            pass
    return_code = process.wait()
    if return_code != 0:
        raise SystemExit(f"FFmpeg failed with exit code {return_code}")
    print(f"Wrote {args.output} ({args.output.stat().st_size} bytes)")
    print(f"Wrote {args.poster} ({args.poster.stat().st_size} bytes)")


def main() -> None:
    args = parse_args()
    Image, ImageDraw, ImageFont = require_pillow()
    ffmpeg = find_ffmpeg(args.ffmpeg)
    metrics = read_report(args.report.resolve())
    encode(Image, ImageDraw, ImageFont, args, metrics, ffmpeg)


if __name__ == "__main__":
    main()
