import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
DEFAULT_PRED_JSON = ROOT / "predict_result" / "MCI_model" / "dual_prior.json"
DEFAULT_DATA_FOLDER = ROOT.parent / "data" / "coding" / "datasets" / "LEVIR-MCI-dataset" / "images"
DEFAULT_TOKEN_FOLDER = ROOT / "data" / "LEVIR_MCI" / "tokens"

GRID_LABELS = {
    (0, 0): "top left corner",
    (0, 1): "top",
    (0, 2): "top right corner",
    (1, 0): "left",
    (1, 1): "center",
    (1, 2): "right",
    (2, 0): "lower left corner",
    (2, 1): "lower",
    (2, 2): "lower right corner",
}
LOC_LABELS = list(GRID_LABELS.values())
SPECIAL_TOKENS = {"<START>", "<END>", "<NULL>"}


def load_records(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]


def normalize_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize_text(text):
    return normalize_text(text).split()


def clean_caption_tokens(tokens):
    return [tok for tok in tokens if tok not in SPECIAL_TOKENS]


def load_reference_captions(token_folder, image_name):
    token_path = Path(token_folder) / f"{Path(image_name).stem}.txt"
    if not token_path.exists():
        return []
    with token_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    captions = []
    for tokens in rows:
        captions.append(" ".join(clean_caption_tokens(tokens)).strip())
    return captions


def load_label_mask(data_folder, split, image_name):
    label_path = Path(data_folder) / split / "label" / image_name
    mask = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Label mask not found: {label_path}")
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mask = mask.astype(np.uint8)
    out = mask.copy()
    if out.max() > 2:
        out = np.zeros_like(mask, dtype=np.uint8)
        out[mask == 128] = 1
        out[mask == 255] = 2
    return out


def count_objects(mask, class_id, min_area=5):
    binary = (mask == class_id).astype(np.uint8)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    return sum(1 for idx in range(1, num_labels) if stats[idx, cv2.CC_STAT_AREA] > min_area)


def grid_locations(mask, class_id, min_pixels=5):
    binary = mask == class_id
    if not np.any(binary):
        return ["No change"]
    height, width = binary.shape
    locations = []
    for row in range(3):
        y0 = round(row * height / 3)
        y1 = round((row + 1) * height / 3)
        for col in range(3):
            x0 = round(col * width / 3)
            x1 = round((col + 1) * width / 3)
            if int(binary[y0:y1, x0:x1].sum()) > min_pixels:
                locations.append(GRID_LABELS[(row, col)])
    return locations or ["No change"]


def prior_from_mask(mask, min_area=5, min_pixels=5):
    return {
        "road_count": count_objects(mask, 1, min_area=min_area),
        "building_count": count_objects(mask, 2, min_area=min_area),
        "road_locations": grid_locations(mask, 1, min_pixels=min_pixels),
        "building_locations": grid_locations(mask, 2, min_pixels=min_pixels),
    }


def location_set(value):
    if value is None:
        return set()
    if isinstance(value, str):
        values = [value]
    else:
        values = list(value)
    joined = ",".join(str(v).lower() for v in values)
    if "no change" in joined:
        return set()
    return {label for label in LOC_LABELS if label in joined}


def corpus_bleu(preds, refs_list, max_n):
    clipped = [0] * max_n
    totals = [0] * max_n
    pred_len = 0
    ref_len = 0

    for pred, refs in zip(preds, refs_list):
        pred_tokens = tokenize_text(pred)
        ref_tokens_list = [tokenize_text(ref) for ref in refs if ref]
        if not ref_tokens_list:
            continue

        pred_len += len(pred_tokens)
        ref_lens = [len(ref_tokens) for ref_tokens in ref_tokens_list]
        ref_len += min(ref_lens, key=lambda length: (abs(length - len(pred_tokens)), length))

        for n in range(1, max_n + 1):
            pred_counts = Counter(tuple(pred_tokens[i:i + n]) for i in range(max(0, len(pred_tokens) - n + 1)))
            max_ref_counts = Counter()
            for ref_tokens in ref_tokens_list:
                ref_counts = Counter(tuple(ref_tokens[i:i + n]) for i in range(max(0, len(ref_tokens) - n + 1)))
                for gram, count in ref_counts.items():
                    max_ref_counts[gram] = max(max_ref_counts[gram], count)
            clipped[n - 1] += sum(min(count, max_ref_counts[gram]) for gram, count in pred_counts.items())
            totals[n - 1] += sum(pred_counts.values())

    if pred_len == 0:
        return 0.0
    bp = 1.0 if pred_len > ref_len else math.exp(1 - ref_len / pred_len) if pred_len else 0.0
    precisions = [(clipped[i] + 1) / (totals[i] + 1) for i in range(max_n)]
    return bp * math.exp(sum(math.log(p) for p in precisions) / max_n)


def caption_metrics(rows):
    preds = [row["pred_caption"] for row in rows]
    refs = [row["ref_captions"] for row in rows]
    exact = 0
    valid = 0
    for pred, row_refs in zip(preds, refs):
        normalized_pred = normalize_text(pred)
        normalized_refs = {normalize_text(ref) for ref in row_refs}
        if normalized_refs:
            valid += 1
            exact += normalized_pred in normalized_refs
    return {
        "n": len(rows),
        "valid": valid,
        "exact_match": exact / valid if valid else 0.0,
        "bleu_1": corpus_bleu(preds, refs, 1),
        "bleu_2": corpus_bleu(preds, refs, 2),
        "bleu_3": corpus_bleu(preds, refs, 3),
        "bleu_4": corpus_bleu(preds, refs, 4),
    }


def count_metrics(rows, key):
    total = len(rows)
    exact = 0
    abs_error = 0
    squared_error = 0
    pairs = []
    for row in rows:
        pred = int(row["pred_prior"].get(key, 0) or 0)
        ref = int(row["gt_prior"].get(key, 0) or 0)
        pairs.append({"image": row["image"], "pred": pred, "gt": ref})
        exact += pred == ref
        err = pred - ref
        abs_error += abs(err)
        squared_error += err * err
    return {
        "n": total,
        "exact_accuracy": exact / total if total else 0.0,
        "mae": abs_error / total if total else 0.0,
        "rmse": math.sqrt(squared_error / total) if total else 0.0,
        "pairs": pairs,
    }


def location_metrics(rows, key):
    total = len(rows)
    exact = 0
    tp = fp = fn = 0
    no_change_total = 0
    no_change_correct = 0
    for row in rows:
        pred = location_set(row["pred_prior"].get(key, []))
        ref = location_set(row["gt_prior"].get(key, []))
        exact += pred == ref
        tp += len(pred & ref)
        fp += len(pred - ref)
        fn += len(ref - pred)
        if not ref:
            no_change_total += 1
            no_change_correct += not pred
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": total,
        "exact_match": exact / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "no_change_accuracy": no_change_correct / no_change_total if no_change_total else None,
    }


def binary_change_metrics(rows):
    total = len(rows)
    correct = 0
    confusion = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for row in rows:
        pred = int(row["pred_prior"].get("road_count", 0) or 0) > 0 or int(row["pred_prior"].get("building_count", 0) or 0) > 0
        ref = int(row["gt_prior"].get("road_count", 0) or 0) > 0 or int(row["gt_prior"].get("building_count", 0) or 0) > 0
        correct += pred == ref
        if pred and ref:
            confusion["tp"] += 1
        elif not pred and not ref:
            confusion["tn"] += 1
        elif pred and not ref:
            confusion["fp"] += 1
        else:
            confusion["fn"] += 1
    return {
        "n": total,
        "accuracy": correct / total if total else 0.0,
        "confusion": confusion,
    }


def draw_bar_chart(values, path, title):
    width, height = 1100, 620
    margin_left, margin_top, margin_bottom = 120, 80, 120
    chart_w = width - margin_left - 60
    chart_h = height - margin_top - margin_bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    title_font = ImageFont.load_default()

    draw.text((margin_left, 28), title, fill=(20, 20, 20), font=title_font)
    draw.line((margin_left, margin_top, margin_left, margin_top + chart_h), fill=(80, 80, 80), width=2)
    draw.line((margin_left, margin_top + chart_h, margin_left + chart_w, margin_top + chart_h), fill=(80, 80, 80), width=2)

    for i in range(6):
        y = margin_top + chart_h - int(chart_h * i / 5)
        draw.line((margin_left - 6, y, margin_left + chart_w, y), fill=(225, 225, 225), width=1)
        draw.text((margin_left - 55, y - 7), f"{i / 5:.1f}", fill=(70, 70, 70), font=font)

    if not values:
        image.save(path)
        return

    bar_gap = 18
    bar_w = max(28, int((chart_w - bar_gap * (len(values) + 1)) / len(values)))
    colors = [(31, 119, 180), (44, 160, 44), (255, 127, 14), (214, 39, 40), (148, 103, 189), (23, 162, 184), (100, 100, 100)]
    for idx, (label, value) in enumerate(values):
        x0 = margin_left + bar_gap + idx * (bar_w + bar_gap)
        x1 = x0 + bar_w
        value = max(0.0, min(1.0, float(value)))
        y1 = margin_top + chart_h
        y0 = y1 - int(chart_h * value)
        draw.rectangle((x0, y0, x1, y1), fill=colors[idx % len(colors)])
        draw.text((x0, y0 - 18), f"{value:.3f}", fill=(30, 30, 30), font=font)
        label_lines = label.split(" ")
        for line_idx, line in enumerate(label_lines):
            draw.text((x0, y1 + 12 + line_idx * 14), line, fill=(45, 45, 45), font=font)

    image.save(path)


def draw_confusion_matrix(confusion, path):
    width, height = 720, 560
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((60, 30), "Binary Change Confusion Matrix", fill=(20, 20, 20), font=font)
    x0, y0, cell = 190, 120, 150
    labels = [["TN", confusion["tn"]], ["FP", confusion["fp"]], ["FN", confusion["fn"]], ["TP", confusion["tp"]]]
    max_val = max(confusion.values()) if confusion else 1
    positions = [(0, 0), (1, 0), (0, 1), (1, 1)]
    for (col, row), (name, value) in zip(positions, labels):
        intensity = int(235 - 160 * (value / max_val if max_val else 0))
        color = (intensity, intensity + 10 if intensity < 245 else 245, 255)
        left = x0 + col * cell
        top = y0 + row * cell
        draw.rectangle((left, top, left + cell, top + cell), fill=color, outline=(70, 70, 70), width=2)
        draw.text((left + 55, top + 50), name, fill=(20, 20, 20), font=font)
        draw.text((left + 55, top + 78), str(value), fill=(20, 20, 20), font=font)
    draw.text((x0, y0 - 30), "Pred: No Change", fill=(40, 40, 40), font=font)
    draw.text((x0 + cell, y0 - 30), "Pred: Change", fill=(40, 40, 40), font=font)
    draw.text((50, y0 + 55), "GT: No Change", fill=(40, 40, 40), font=font)
    draw.text((50, y0 + cell + 55), "GT: Change", fill=(40, 40, 40), font=font)
    image.save(path)


def main():
    parser = argparse.ArgumentParser(description="Evaluate and visualize Dual-Branch dual_prior predictions.")
    parser.add_argument("--pred-json", default=str(DEFAULT_PRED_JSON))
    parser.add_argument("--data-folder", default=str(DEFAULT_DATA_FOLDER))
    parser.add_argument("--token-folder", default=str(DEFAULT_TOKEN_FOLDER))
    parser.add_argument("--split", default="test")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--min-area", type=int, default=5)
    parser.add_argument("--min-pixels", type=int, default=5)
    args = parser.parse_args()

    pred_json = Path(args.pred_json)
    records = load_records(pred_json)
    out_dir = Path(args.out_dir) if args.out_dir else pred_json.parent / "metric_visualization"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    errors = []
    for record in records:
        if record.get("error"):
            errors.append({"image": record.get("image"), "error": record.get("error")})
            continue
        image_name = record.get("image") or record.get("name")
        try:
            gt_mask = load_label_mask(args.data_folder, args.split, image_name)
            gt_prior = prior_from_mask(gt_mask, min_area=args.min_area, min_pixels=args.min_pixels)
            ref_captions = load_reference_captions(args.token_folder, image_name)
            rows.append({
                "image": image_name,
                "pred_prior": record.get("dual_prior", {}),
                "gt_prior": gt_prior,
                "pred_caption": record.get("dual_prior", {}).get("global_caption", ""),
                "ref_captions": ref_captions,
            })
        except Exception as exc:
            errors.append({"image": image_name, "error": repr(exc)})

    metrics = {
        "source": str(pred_json),
        "evaluated": len(rows),
        "skipped": len(errors),
        "caption": caption_metrics(rows),
        "binary_change": binary_change_metrics(rows),
        "road_count": count_metrics(rows, "road_count"),
        "building_count": count_metrics(rows, "building_count"),
        "road_locations": location_metrics(rows, "road_locations"),
        "building_locations": location_metrics(rows, "building_locations"),
        "errors": errors,
    }

    metrics_json = out_dir / "metrics.json"
    with metrics_json.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    per_sample_json = out_dir / "per_sample.json"
    with per_sample_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    summary_values = [
        ("Caption BLEU-4", metrics["caption"]["bleu_4"]),
        ("Caption Exact", metrics["caption"]["exact_match"]),
        ("Binary Acc", metrics["binary_change"]["accuracy"]),
        ("Road Count Acc", metrics["road_count"]["exact_accuracy"]),
        ("Building Count Acc", metrics["building_count"]["exact_accuracy"]),
        ("Road Loc F1", metrics["road_locations"]["f1"]),
        ("Building Loc F1", metrics["building_locations"]["f1"]),
    ]
    draw_bar_chart(summary_values, out_dir / "summary_metrics.png", "Dual Prior vs Ground Truth Metrics")
    draw_confusion_matrix(metrics["binary_change"]["confusion"], out_dir / "binary_change_confusion.png")

    print(f"Evaluated rows: {len(rows)}")
    print(f"Skipped rows: {len(errors)}")
    print(f"Saved metrics: {metrics_json}")
    print(f"Saved per-sample comparison: {per_sample_json}")
    print(f"Saved chart: {out_dir / 'summary_metrics.png'}")
    print(f"Saved confusion matrix: {out_dir / 'binary_change_confusion.png'}")
    print("Key metrics:")
    for label, value in summary_values:
        print(f"  {label}: {value:.5f}")


if __name__ == "__main__":
    main()
