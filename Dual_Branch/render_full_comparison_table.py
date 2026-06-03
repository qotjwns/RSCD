import argparse
import csv
import json
import textwrap
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
    ("Caption*", "BLEU-4"),
    ("", "METEOR"),
    ("", "ROUGE-L"),
    ("", "CIDEr"),
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
    ("Caption*", "BLEU-4"): 65.08,
    ("Caption*", "METEOR"): 42.06,
    ("Caption*", "ROUGE-L"): 76.95,
    ("Caption*", "CIDEr"): 142.83,
}

LOWER_IS_BETTER = {("Counting", "Road MAE"), ("Counting", "Build MAE")}
CAPTION_TASKS = {"Caption*"}


def font(size, bold=False, italic=False):
    candidates = []
    if italic:
        candidates.extend([
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf",
            "/usr/share/fonts/dejavu/DejaVuSerif-Italic.ttf",
        ])
    elif bold:
        candidates.extend([
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
            "/usr/share/fonts/dejavu/DejaVuSerif-Bold.ttf",
        ])
    else:
        candidates.extend([
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
            "/usr/share/fonts/dejavu/DejaVuSerif.ttf",
        ])
    candidates.extend([
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ])
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def safe_div(num, den):
    return num / den if den else 0.0


def loc_set(value):
    values = value if isinstance(value, list) else [value]
    joined = ",".join(str(v).lower() for v in values)
    if "no change" in joined:
        return set()
    return {label for label in LOC_LABELS if label in joined}


def binary_flag(prior):
    road = int(prior.get("road_count", 0) or 0)
    build = int(prior.get("building_count", 0) or 0)
    return road > 0 or build > 0


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
        return None
    total = 0.0
    for row in rows:
        pred = int(row["pred_prior"].get(key, 0) or 0)
        gt = int(row["gt_prior"].get(key, 0) or 0)
        total += abs(pred - gt)
    return total / len(rows)


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


def compute_caption_coco(rows):
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge

    refs = {}
    hyps = {}
    for idx, row in enumerate(rows):
        row_refs = [str(ref).strip() for ref in row.get("ref_captions", []) if str(ref).strip()]
        refs[idx] = row_refs or [""]
        hyps[idx] = [str(row.get("pred_caption", "")).strip()]

    values = {}
    scorers = [
        (Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE-L"),
        (Cider(), "CIDEr"),
    ]
    for scorer, names in scorers:
        score, _ = scorer.compute_score(refs, hyps)
        if isinstance(names, list):
            for name, value in zip(names, score):
                values[name] = value * 100.0
        else:
            values[names] = score * 100.0
    return values


def compute_model_values(per_sample_path, skip_caption=False):
    rows = json.load(open(per_sample_path, "r", encoding="utf-8"))
    binary = compute_binary(rows)
    road_loc = compute_location(rows, "road_locations")
    build_loc = compute_location(rows, "building_locations")
    values = {
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
    if not skip_caption:
        caption = compute_caption_coco(rows)
        values.update({("Caption*", metric): caption.get(metric) for metric in ["BLEU-4", "METEOR", "ROUGE-L", "CIDEr"]})
    return values, len(rows)


def is_lower_better(task, metric):
    return (task, metric) in LOWER_IS_BETTER


def best_columns(task, metric, values):
    valid = [(idx, value) for idx, value in enumerate(values) if value is not None]
    if not valid:
        return set()
    target = min(value for _, value in valid) if is_lower_better(task, metric) else max(value for _, value in valid)
    return {idx for idx, value in valid if abs(value - target) < 1e-12}


def fmt(task, value):
    if value is None:
        return "-"
    if task in CAPTION_TASKS:
        return f"{value:.2f}"
    return f"{value:.4f}"


def section_end_indices():
    ends = set()
    for idx, _ in enumerate(ROWS):
        next_is_new = idx + 1 < len(ROWS) and ROWS[idx + 1][0]
        if next_is_new or idx == len(ROWS) - 1:
            ends.add(idx)
    return ends


def text_size(draw, text, draw_font):
    bbox = draw.textbbox((0, 0), text, font=draw_font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def center_text(draw, box, text, draw_font, fill):
    x0, y0, x1, y1 = box
    width, height = text_size(draw, text, draw_font)
    draw.text((x0 + (x1 - x0 - width) / 2, y0 + (y1 - y0 - height) / 2 - 2), text, font=draw_font, fill=fill)


def left_text(draw, box, text, draw_font, fill):
    x0, y0, _, y1 = box
    _, height = text_size(draw, text, draw_font)
    draw.text((x0, y0 + (y1 - y0 - height) / 2 - 2), text, font=draw_font, fill=fill)


def draw_wrapped(draw, xy, text, draw_font, fill, width_chars, line_gap=5):
    x, y = xy
    for line in textwrap.wrap(text, width=width_chars):
        draw.text((x, y), line, font=draw_font, fill=fill)
        _, h = text_size(draw, line, draw_font)
        y += h + line_gap
    return y


def render_table(paper, reproduce, fft, out_path, title, subtitle):
    col_w = [125, 205, 265, 230, 225]
    margin_x = 42
    title_h = 120
    header_h = 50
    row_h = 42
    foot_h = 112
    width = margin_x * 2 + sum(col_w)
    table_h = header_h + row_h * len(ROWS)
    height = title_h + table_h + foot_h + 30

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    black = (25, 25, 25)
    muted = (70, 70, 70)
    line = (88, 88, 88)
    light_line = (185, 185, 185)
    green = (23, 124, 48)

    f_title = font(25, bold=True)
    f_sub = font(16, italic=True)
    f_head = font(15, bold=True)
    f_group = font(15, bold=True)
    f_cell = font(15)
    f_bold = font(15, bold=True)
    f_note = font(12)

    center_text(draw, (0, 20, width, 58), title, f_title, black)
    center_text(draw, (0, 66, width, 96), subtitle, f_sub, muted)

    x_edges = [margin_x]
    for w in col_w:
        x_edges.append(x_edges[-1] + w)
    y = title_h

    draw.line((margin_x, y, width - margin_x, y), fill=black, width=3)
    headers = ["Task", "Metric", "Paper Full ChangeVG", "Reproduce", "FFT"]
    for idx, label in enumerate(headers):
        center_text(draw, (x_edges[idx], y, x_edges[idx + 1], y + header_h), label, f_head, black)
    draw.line((margin_x, y + header_h, width - margin_x, y + header_h), fill=line, width=1)

    current_task = None
    section_ends = section_end_indices()
    for row_idx, (task_label, metric) in enumerate(ROWS):
        if task_label:
            current_task = task_label
        task = current_task
        y0 = y + header_h + row_idx * row_h
        y1 = y0 + row_h
        if task_label:
            left_text(draw, (x_edges[0] + 8, y0, x_edges[1], y1), task_label, f_group, black)
        left_text(draw, (x_edges[1] + 6, y0, x_edges[2], y1), metric, f_cell, black)

        row_values = [paper.get((task, metric)), reproduce.get((task, metric)), fft.get((task, metric))]
        best = best_columns(task, metric, row_values)
        paper_value = row_values[0]
        for value_idx, value in enumerate(row_values):
            col_idx = value_idx + 2
            is_best = value_idx in best
            draw_font = f_bold if is_best else f_cell
            fill = black
            if value_idx == 2 and is_best and paper_value is not None and value is not None:
                beats_paper = value < paper_value if is_lower_better(task, metric) else value > paper_value
                if beats_paper:
                    fill = green
            center_text(draw, (x_edges[col_idx], y0, x_edges[col_idx + 1], y1), fmt(task, value), draw_font, fill)

        if row_idx in section_ends:
            draw.line((margin_x, y1, width - margin_x, y1), fill=light_line if row_idx < len(ROWS) - 1 else black, width=1 if row_idx < len(ROWS) - 1 else 3)

    note_y = y + table_h + 22
    note = (
        "Paper baselines are the Paper Full ChangeVG values from the reference table. "
        "Reproduce uses predict_result/MCI_model; FFT uses predict_result/MCI_model_FFT. "
        "Caption* metrics are COCO caption metrics computed from per_sample captions and shown as percentages. "
        "Bold = best value in the row. Green bold = FFT beats Paper Full ChangeVG."
    )
    draw_wrapped(draw, (margin_x + 4, note_y), note, f_note, black, width_chars=150)

    image.save(out_path)


def write_csv(paper, reproduce, fft, out_path):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Task", "Metric", "Paper Full ChangeVG", "Reproduce", "FFT"])
        current_task = None
        for task_label, metric in ROWS:
            if task_label:
                current_task = task_label
            task = current_task
            writer.writerow([
                task_label,
                metric,
                fmt(task, paper.get((task, metric))),
                fmt(task, reproduce.get((task, metric))),
                fmt(task, fft.get((task, metric))),
            ])


def main():
    parser = argparse.ArgumentParser(description="Render Paper/Reproduce/FFT full comparison table.")
    parser.add_argument("--reproduce-per-sample", default="predict_result/MCI_model/metric_visualization/per_sample.json")
    parser.add_argument("--fft-per-sample", default="predict_result/MCI_model_FFT/metric_visualization/per_sample.json")
    parser.add_argument("--out-dir", default="predict_result/full_comparison")
    parser.add_argument("--title", default="Dual Branch on LEVIR-MCI - Full Comparison")
    parser.add_argument("--subtitle", default="Paper Full ChangeVG vs MCI_model reproduce vs MCI_model_FFT")
    parser.add_argument("--skip-caption", action="store_true", help="Skip COCO caption metric computation.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reproduce, reproduce_n = compute_model_values(args.reproduce_per_sample, skip_caption=args.skip_caption)
    fft, fft_n = compute_model_values(args.fft_per_sample, skip_caption=args.skip_caption)

    png_path = out_dir / "paper_reproduce_fft_comparison.png"
    csv_path = out_dir / "paper_reproduce_fft_comparison.csv"
    render_table(PAPER_VALUES, reproduce, fft, png_path, args.title, args.subtitle)
    write_csv(PAPER_VALUES, reproduce, fft, csv_path)

    print(f"Loaded reproduce rows: {reproduce_n}")
    print(f"Loaded FFT rows: {fft_n}")
    print(f"Saved table PNG: {png_path}")
    print(f"Saved table CSV: {csv_path}")


if __name__ == "__main__":
    main()
