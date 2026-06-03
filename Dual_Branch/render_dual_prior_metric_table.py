import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

LOC_LABELS = [
    "top left corner",
    "top",
    "top right corner",
    "left",
    "center",
    "right",
    "lower left corner",
    "lower",
    "lower right corner",
]
PAPER_VALUES = {
    ("Binary", "Accuracy"): 0.9460,
    ("Binary", "Precision"): 0.9549,
    ("Binary", "Recall"): 0.9401,
    ("Binary", "F1"): 0.9474,
    ("Counting", "Road MAE"): 0.1560,
    ("Counting", "Build MAE"): 0.8020,
    ("Loc-Road", "Example Acc"): 0.7865,
    ("Loc-Road", "Micro P"): 0.7850,
    ("Loc-Road", "Micro R"): 0.7595,
    ("Loc-Road", "Micro F1"): 0.7720,
    ("Loc-Road", "Subset Acc"): 0.7330,
    ("Loc-Build", "Example Acc"): 0.8813,
    ("Loc-Build", "Micro P"): 0.9068,
    ("Loc-Build", "Micro R"): 0.9208,
    ("Loc-Build", "Micro F1"): 0.9137,
    ("Loc-Build", "Subset Acc"): 0.7220,
}
LOWER_IS_BETTER = {("Counting", "Road MAE"), ("Counting", "Build MAE")}
ROWS = [
    ("Binary", "Accuracy"),
    ("", "Precision"),
    ("", "Recall"),
    ("", "F1"),
    ("Counting", "Road MAE"),
    ("", "Build MAE"),
    ("Loc-Road", "Example Acc"),
    ("", "Micro P"),
    ("", "Micro R"),
    ("", "Micro F1"),
    ("", "Subset Acc"),
    ("Loc-Build", "Example Acc"),
    ("", "Micro P"),
    ("", "Micro R"),
    ("", "Micro F1"),
    ("", "Subset Acc"),
    ("Caption", "Exact Match"),
    ("", "BLEU-1"),
    ("", "BLEU-2"),
    ("", "BLEU-3"),
    ("", "BLEU-4"),
]


def font(size, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def loc_set(value):
    values = value if isinstance(value, list) else [value]
    joined = ",".join(str(v).lower() for v in values)
    if "no change" in joined:
        return set()
    return {label for label in LOC_LABELS if label in joined}


def safe_div(num, den):
    return num / den if den else 0.0


def binary_flag(prior):
    return int(prior.get("road_count", 0) or 0) > 0 or int(prior.get("building_count", 0) or 0) > 0


def compute_binary(rows):
    tp = tn = fp = fn = 0
    for row in rows:
        pred = binary_flag(row["pred_prior"])
        gt = binary_flag(row["gt_prior"])
        if pred and gt:
            tp += 1
        elif not pred and not gt:
            tn += 1
        elif pred and not gt:
            fp += 1
        else:
            fn += 1
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {
        "Accuracy": safe_div(tp + tn, tp + tn + fp + fn),
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }


def compute_count(rows, key):
    if not rows:
        return 0.0
    return sum(abs(int(row["pred_prior"].get(key, 0) or 0) - int(row["gt_prior"].get(key, 0) or 0)) for row in rows) / len(rows)


def compute_location(rows, key):
    exact = 0
    jaccard_total = 0.0
    tp = fp = fn = 0
    for row in rows:
        pred = loc_set(row["pred_prior"].get(key, []))
        gt = loc_set(row["gt_prior"].get(key, []))
        exact += pred == gt
        union = pred | gt
        jaccard_total += 1.0 if not union else len(pred & gt) / len(union)
        tp += len(pred & gt)
        fp += len(pred - gt)
        fn += len(gt - pred)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    total = len(rows)
    return {
        "Example Acc": safe_div(jaccard_total, total),
        "Micro P": precision,
        "Micro R": recall,
        "Micro F1": f1,
        "Subset Acc": safe_div(exact, total),
    }


def load_metrics(path):
    if not path:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def add_caption_values(ours, metrics):
    caption = metrics.get("caption", {})
    metric_keys = {
        "Exact Match": "exact_match",
        "BLEU-1": "bleu_1",
        "BLEU-2": "bleu_2",
        "BLEU-3": "bleu_3",
        "BLEU-4": "bleu_4",
    }
    for metric, key in metric_keys.items():
        value = caption.get(key)
        ours[("Caption", metric)] = float(value) if value is not None else None


def compute_table_values(rows, metrics=None):
    binary = compute_binary(rows)
    road_loc = compute_location(rows, "road_locations")
    build_loc = compute_location(rows, "building_locations")
    ours = {
        ("Binary", "Accuracy"): binary["Accuracy"],
        ("Binary", "Precision"): binary["Precision"],
        ("Binary", "Recall"): binary["Recall"],
        ("Binary", "F1"): binary["F1"],
        ("Counting", "Road MAE"): compute_count(rows, "road_count"),
        ("Counting", "Build MAE"): compute_count(rows, "building_count"),
        ("Loc-Road", "Example Acc"): road_loc["Example Acc"],
        ("Loc-Road", "Micro P"): road_loc["Micro P"],
        ("Loc-Road", "Micro R"): road_loc["Micro R"],
        ("Loc-Road", "Micro F1"): road_loc["Micro F1"],
        ("Loc-Road", "Subset Acc"): road_loc["Subset Acc"],
        ("Loc-Build", "Example Acc"): build_loc["Example Acc"],
        ("Loc-Build", "Micro P"): build_loc["Micro P"],
        ("Loc-Build", "Micro R"): build_loc["Micro R"],
        ("Loc-Build", "Micro F1"): build_loc["Micro F1"],
        ("Loc-Build", "Subset Acc"): build_loc["Subset Acc"],
    }
    add_caption_values(ours, metrics or {})
    return ours


def is_better(task, metric, a, b):
    if a is None or b is None:
        return False
    if (task, metric) in LOWER_IS_BETTER:
        return a < b
    return a > b


def fmt(value):
    if value is None:
        return "-"
    return f"{value:.4f}"


def section_end_indices():
    ends = set()
    for idx, _ in enumerate(ROWS):
        next_is_new_task = idx + 1 < len(ROWS) and ROWS[idx + 1][0]
        if next_is_new_task or idx == len(ROWS) - 1:
            ends.add(idx)
    return ends


def center_text(draw, box, text, draw_font, fill):
    x0, y0, x1, y1 = box
    bbox = draw.textbbox((0, 0), text, font=draw_font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text((x0 + (x1 - x0 - tw) / 2, y0 + (y1 - y0 - th) / 2 - 1), text, font=draw_font, fill=fill)


def render_table(ours, out_path, model_name):
    cols = ["Task", "Metric", "Paper Full ChangeVG", model_name]
    col_w = [135, 205, 270, 240]
    row_h = 41
    header_h = 48
    title_h = 18
    margin = 36
    width = sum(col_w) + margin * 2
    height = title_h + header_h + row_h * len(ROWS) + margin * 2
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    f_header = font(15, bold=True)
    f_cell = font(14)
    f_bold = font(14, bold=True)

    x_edges = [margin]
    for w in col_w:
        x_edges.append(x_edges[-1] + w)
    y = margin + title_h

    header_fill = (231, 237, 243)
    task_fill = (248, 250, 252)
    grid = (120, 120, 120)
    light_grid = (168, 168, 168)
    green = (54, 142, 61)
    black = (30, 30, 30)

    draw.rectangle((margin, y, width - margin, y + header_h), fill=header_fill, outline=grid)
    for idx, label in enumerate(cols):
        center_text(draw, (x_edges[idx], y, x_edges[idx + 1], y + header_h), label, f_header, black)
    for x in x_edges:
        draw.line((x, y, x, y + header_h + row_h * len(ROWS)), fill=grid, width=1)
    draw.line((margin, y, width - margin, y), fill=grid, width=2)
    draw.line((margin, y + header_h, width - margin, y + header_h), fill=grid, width=1)

    current_task = None
    display_task = None
    section_ends = section_end_indices()
    for row_idx, (task_label, metric) in enumerate(ROWS):
        if task_label:
            current_task = task_label
            display_task = task_label
        task = current_task
        y0 = y + header_h + row_idx * row_h
        y1 = y0 + row_h
        if task_label:
            draw.rectangle((x_edges[0], y0, x_edges[1], y1), fill=task_fill)
            center_text(draw, (x_edges[0], y0, x_edges[1], y1), display_task, f_bold, black)
        center_text(draw, (x_edges[1], y0, x_edges[2], y1), metric, f_cell, black)

        paper_value = PAPER_VALUES.get((task, metric))
        ours_value = ours.get((task, metric))
        has_pair = paper_value is not None and ours_value is not None
        tied = has_pair and abs(paper_value - ours_value) < 1e-12
        paper_best = tied or is_better(task, metric, paper_value, ours_value)
        ours_best = tied or is_better(task, metric, ours_value, paper_value)

        center_text(
            draw,
            (x_edges[2], y0, x_edges[3], y1),
            fmt(paper_value),
            f_bold if paper_best else f_cell,
            black,
        )
        center_text(
            draw,
            (x_edges[3], y0, x_edges[4], y1),
            fmt(ours_value),
            f_bold if ours_best else f_cell,
            green if ours_best and is_better(task, metric, ours_value, paper_value) else black,
        )

        line_color = grid if row_idx in section_ends else light_grid
        line_width = 2 if row_idx in section_ends else 1
        draw.line((margin, y1, width - margin, y1), fill=line_color, width=line_width)

    draw.rectangle((margin, y, width - margin, y + header_h + row_h * len(ROWS)), outline=grid, width=1)
    image.save(out_path)


def write_csv(ours, out_path, model_name):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Task", "Metric", "Paper Full ChangeVG", model_name])
        current_task = None
        for task_label, metric in ROWS:
            if task_label:
                current_task = task_label
            task = current_task
            writer.writerow([task_label, metric, fmt(PAPER_VALUES.get((task, metric))), fmt(ours.get((task, metric)))])


def main():
    parser = argparse.ArgumentParser(description="Render paper-style Dual Prior metric table.")
    parser.add_argument("--per-sample-json", default="predict_result/MCI_model/metric_visualization/per_sample.json")
    parser.add_argument("--metrics-json", default=None)
    parser.add_argument("--out-dir", default="predict_result/MCI_model/metric_visualization")
    parser.add_argument("--model-name", default="MCI_model (reproduce)")
    args = parser.parse_args()

    per_sample_path = Path(args.per_sample_json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = json.load(open(per_sample_path, "r", encoding="utf-8"))
    metrics_path = Path(args.metrics_json) if args.metrics_json else per_sample_path.with_name("metrics.json")
    metrics = load_metrics(metrics_path)
    ours = compute_table_values(rows, metrics)

    png_path = out_dir / "paper_style_metrics_table.png"
    csv_path = out_dir / "paper_style_metrics_table.csv"
    render_table(ours, png_path, args.model_name)
    write_csv(ours, csv_path, args.model_name)

    print(f"Loaded rows: {len(rows)}")
    print(f"Loaded metrics: {metrics_path if metrics else 'not found'}")
    print(f"Saved table PNG: {png_path}")
    print(f"Saved table CSV: {csv_path}")


if __name__ == "__main__":
    main()
