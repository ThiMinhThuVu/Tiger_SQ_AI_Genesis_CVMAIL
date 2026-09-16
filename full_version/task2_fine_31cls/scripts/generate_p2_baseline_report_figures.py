#!/usr/bin/env python3
"""Generate reproducible 300-DPI figures for the P0 versus P2 report.

The workspace Python has Pillow but no plotting stack, so this module deliberately
uses Pillow only. Charts preserve a zero baseline where absolute bars are shown and
use signed deltas for small between-model differences.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[3]
TASK = ROOT / "full_version/task2_fine_31cls"
ARTIFACTS = TASK / "artifacts"
P0_NATIVE = ARTIFACTS / "p0_baseline_lock_v2/canonical_native_oof_metrics.json"
BASELINE_GRID = ARTIFACTS / "seed_2026/test_oof/folds"
P2_GRID = ARTIFACTS / "p2_hr_oof/folds"
AUDIT = ARTIFACTS / "p2_resolution_audit_v1/p2_resolution_audit.json"
OUT = ARTIFACTS / "p2_hr_oof/report_assets"

W, H = 2070, 1080
MARGIN = 105
BASELINE = "#0077BB"
P2 = "#EE7733"
POSITIVE = "#009988"
NEGATIVE = "#CC3311"
NEUTRAL = "#667085"
INK = "#1D2939"
GRID = "#D0D5DD"
LIGHT_BLUE = "#E7F2F8"
LIGHT_ORANGE = "#FFF0E5"
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(FONT_DIR / name), size=size)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def canvas(height: int = H):
    image = Image.new("RGB", (W, height), "white")
    return image, ImageDraw.Draw(image)


def save(image: Image.Image, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    image.save(OUT / name, dpi=(300, 300), quality=95)


def text_center(draw, xy, value, fnt, fill=INK):
    draw.text(xy, value, font=fnt, fill=fill, anchor="mm", align="center")


def title(draw, value: str, subtitle: str | None = None) -> None:
    draw.text((MARGIN, 58), value, font=font(38, True), fill=INK)
    if subtitle:
        draw.text((MARGIN, 112), subtitle, font=font(24), fill=NEUTRAL)


def round_box(draw, box, outline, fill, heading, lines):
    draw.rounded_rectangle(box, radius=24, fill=fill, outline=outline, width=5)
    x0, y0, x1, y1 = box
    text_center(draw, ((x0 + x1) / 2, y0 + 55), heading, font(31, True), outline)
    for i, line in enumerate(lines):
        text_center(draw, ((x0 + x1) / 2, y0 + 117 + i * 42), line, font(22), INK)


def arrow(draw, start, end, fill=NEUTRAL, width=7):
    draw.line([start, end], fill=fill, width=width)
    x1, y1 = end
    x0, y0 = start
    dx, dy = x1 - x0, y1 - y0
    length = max((dx * dx + dy * dy) ** 0.5, 1)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    size = 24
    points = [
        (x1, y1),
        (x1 - ux * size + px * size * 0.55, y1 - uy * size + py * size * 0.55),
        (x1 - ux * size - px * size * 0.55, y1 - uy * size - py * size * 0.55),
    ]
    draw.polygon(points, fill=fill)


def figure_method() -> None:
    image, draw = canvas(1280)
    title(draw, "Figure 1. Controlled method: P0 baseline versus P2 high resolution")
    round_box(draw, (105, 160, 1965, 410), NEUTRAL, "#F2F4F7",
              "P1 diagnosis → P2 hypothesis",
              ["Small/thin anatomy and missed-small-object errors are major signals",
               "Resolution audit tests whether 512×896 compresses target components"])
    round_box(draw, (105, 555, 900, 910), BASELINE, LIGHT_BLUE,
              "P0 frozen baseline",
              ["512×896 input", "batch 4 × accumulation 1 = effective batch 4",
               "Swin-S Mask2Former", "seed · splits · optimizer · loss frozen"])
    round_box(draw, (1170, 555, 1965, 910), P2, LIGHT_ORANGE,
              "P2 high resolution",
              ["640×1120 input (+25% each dimension)", "batch 2 × accumulation 2 = effective batch 4",
               "same model and training protocol", "single scientific intervention: resolution"])
    round_box(draw, (395, 1050, 1675, 1225), POSITIVE, "#E8F7F4",
              "Matched evaluation gates",
              ["inner validation → held-out fold test → native-resolution 5-fold OOF"])
    arrow(draw, (1035, 410), (500, 545))
    arrow(draw, (1035, 410), (1570, 545))
    arrow(draw, (500, 910), (765, 1040))
    arrow(draw, (1570, 910), (1305, 1040))
    save(image, "figure_01_method_overview.png")


def available_fold_pairs():
    rows = []
    for p2_path in sorted(P2_GRID.glob("fold_*.json")):
        baseline_path = BASELINE_GRID / p2_path.name
        if baseline_path.is_file():
            p2 = load_json(p2_path)
            baseline = load_json(baseline_path)
            rows.append((int(p2["fold"]), baseline, p2))
    return sorted(rows)


def draw_signed_bar_chart(title_text, subtitle, labels, values, filename, unit_label,
                          positive=POSITIVE, negative=NEGATIVE, height=1080):
    image, draw = canvas(height)
    title(draw, title_text, subtitle)
    left, right, top, bottom = 500, W - 130, 220, height - 160
    max_abs = max(max(abs(v) for v in values) * 1.25, 0.1)
    zero = (left + right) / 2
    draw.line((zero, top - 15, zero, bottom + 15), fill=INK, width=4)
    for tick in [-max_abs, -max_abs / 2, 0, max_abs / 2, max_abs]:
        x = zero + tick / max_abs * (right - left) / 2
        draw.line((x, top, x, bottom), fill=GRID, width=2)
        text_center(draw, (x, bottom + 48), f"{tick:+.1f}", font(19), NEUTRAL)
    row_h = (bottom - top) / len(labels)
    for i, (label, value) in enumerate(zip(labels, values)):
        y = top + (i + 0.5) * row_h
        draw.text((left - 30, y), label, font=font(24, i == len(labels) - 1),
                  fill=INK, anchor="rm")
        x = zero + value / max_abs * (right - left) / 2
        color = positive if value >= 0 else negative
        draw.rectangle((min(zero, x), y - 26, max(zero, x), y + 26), fill=color)
        draw.text((x + (12 if value >= 0 else -12), y), f"{value:+.2f}",
                  font=font(22, True), fill=color, anchor="lm" if value >= 0 else "rm")
    text_center(draw, ((left + right) / 2, bottom + 105), unit_label, font(22), INK)
    save(image, filename)


def figure_fold_delta(rows) -> None:
    labels = [f"Fold {fold}" for fold, _, _ in rows]
    values = [(p2["task1_score"] - baseline["task1_score"]) * 100
              for _, baseline, p2 in rows]
    draw_signed_bar_chart(
        f"Figure 2. Held-out test delta by available fold (partial: {len(rows)}/5 folds)",
        "Grid-specific: baseline 512×896; P2 640×1120 — not canonical native-resolution OOF",
        labels, values, "figure_02_partial_test_fold_delta.png",
        "P2 − baseline Task score (percentage points)",
    )


def get_case_task(payload: dict, case: str) -> tuple[float, float, float]:
    dice = float(payload["fine"]["per_case_dice"][case])
    nhd = float(payload["fine"]["per_case_nhd"][case])
    return (dice + 1.0 - nhd) / 2.0, dice, nhd


def figure_c4(rows) -> None:
    labels, values = [], []
    for _, baseline, p2 in rows:
        for case in p2["test_cases"]:
            if case.startswith("center_4_"):
                b_task, _, _ = get_case_task(baseline, case)
                p_task, _, _ = get_case_task(p2, case)
                labels.append(case.replace("center_4_case_", "C4 case "))
                values.append((p_task - b_task) * 100)
    labels.append("C4 mean")
    values.append(sum(values) / len(values))
    draw_signed_bar_chart(
        "Figure 3. Center 4: all currently observed held-out cases improve",
        "n=3 cases; robustness signal, not proof of a center-wide causal effect",
        labels, values, "figure_03_c4_case_delta.png",
        "P2 improvement in Task score (percentage points)", positive=P2,
    )


def figure_resolution_risk() -> None:
    rows = sorted(load_json(AUDIT)["overall"],
                  key=lambda row: row["projected_component_lt16_rate"])
    labels = [row["class_name"] for row in rows]
    values = [row["projected_component_lt16_rate"] * 100 for row in rows]
    image, draw = canvas(1260)
    title(draw, "Figure 4. Resolution-survival audit used to select P2",
          "Decision rule: select high resolution if any target class reaches ≥20% projected components below 16 px")
    left, right, top, bottom = 780, W - 130, 225, 1100
    max_value = 32.0
    for tick in range(0, 31, 5):
        x = left + tick / max_value * (right - left)
        draw.line((x, top, x, bottom), fill=GRID, width=2)
        text_center(draw, (x, bottom + 45), str(tick), font(19), NEUTRAL)
    threshold_x = left + 20 / max_value * (right - left)
    draw.line((threshold_x, top - 10, threshold_x, bottom + 10), fill=INK, width=4)
    draw.text((threshold_x + 10, top - 18), "20% threshold", font=font(19, True), fill=INK, anchor="ls")
    row_h = (bottom - top) / len(rows)
    for i, (label, value) in enumerate(zip(labels, values)):
        y = top + (i + 0.5) * row_h
        draw.text((left - 25, y), label, font=font(21), fill=INK, anchor="rm")
        x = left + value / max_value * (right - left)
        color = NEGATIVE if value >= 20 else BASELINE
        draw.rectangle((left, y - 25, x, y + 25), fill=color)
        draw.text((x + 10, y), f"{value:.1f}%", font=font(20, True), fill=color, anchor="lm")
    text_center(draw, ((left + right) / 2, bottom + 100),
                "Projected components below 16 px at 512×896 (%)", font(22), INK)
    save(image, "figure_04_resolution_risk.png")


def figure_metric_summary(rows) -> None:
    metrics = ["Task", "Dice", "1 − nHD"]
    baseline = [
        sum(b["task1_score"] for _, b, _ in rows) / len(rows),
        sum(b["fine_dice"] for _, b, _ in rows) / len(rows),
        sum(1 - b["fine_nhd"] for _, b, _ in rows) / len(rows),
    ]
    p2 = [
        sum(p["task1_score"] for _, _, p in rows) / len(rows),
        sum(p["fine_dice"] for _, _, p in rows) / len(rows),
        sum(1 - p["fine_nhd"] for _, _, p in rows) / len(rows),
    ]
    image, draw = canvas(1200)
    title(draw, f"Figure 5. Partial matched-fold metric summary ({len(rows)}/5 folds)",
          "Absolute bars start at zero; small numerical differences are labeled directly")
    left, right, top, bottom = 220, W - 120, 220, 1010
    for tick in [0, 0.25, 0.50, 0.75, 1.0]:
        y = bottom - tick * (bottom - top)
        draw.line((left, y, right, y), fill=GRID, width=2)
        draw.text((left - 20, y), f"{tick:.2f}", font=font(19), fill=NEUTRAL, anchor="rm")
    group_w = (right - left) / len(metrics)
    bar_w = 150
    for i, (label, b, p) in enumerate(zip(metrics, baseline, p2)):
        center = left + (i + 0.5) * group_w
        for value, x, color in [(b, center - 90, BASELINE), (p, center + 90, P2)]:
            y = bottom - value * (bottom - top)
            draw.rectangle((x - bar_w / 2, y, x + bar_w / 2, bottom), fill=color)
            text_center(draw, (x, y - 28), f"{value:.4f}", font(20, True), color)
        text_center(draw, (center, bottom + 52), label, font(23, True), INK)
    draw.rectangle((W - 600, 70, W - 560, 110), fill=BASELINE)
    draw.text((W - 545, 90), "Baseline grid", font=font(20), fill=INK, anchor="lm")
    draw.rectangle((W - 330, 70, W - 290, 110), fill=P2)
    draw.text((W - 275, 90), "P2", font=font(20), fill=INK, anchor="lm")
    save(image, "figure_05_partial_metric_summary.png")


def main() -> None:
    _ = load_json(P0_NATIVE)  # Fail early if the frozen baseline is absent.
    rows = available_fold_pairs()
    if not rows:
        raise RuntimeError("No matched baseline/P2 held-out fold artifacts were found")
    figure_method()
    figure_fold_delta(rows)
    figure_c4(rows)
    figure_resolution_risk()
    figure_metric_summary(rows)
    print(f"Generated 5 figures in {OUT} from folds {[fold for fold, _, _ in rows]}")


if __name__ == "__main__":
    main()
