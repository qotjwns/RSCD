import argparse
import json
import re
from collections import defaultdict


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


def load_records(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            records = json.load(f)
        else:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row["_line_no"] = line_no
                records.append(row)
    return records


def normalize_yes_no(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z]+", " ", text).strip()
    if text.startswith("yes") or re.search(r"\byes\b", text):
        return "yes"
    if text.startswith("no") or re.search(r"\bno\b", text):
        return "no"
    return text


def first_int(text):
    match = re.search(r"-?\d+", str(text))
    return int(match.group()) if match else None


def parse_reference_count(text):
    text = str(text).lower()
    if "no change" in text or "none" in text:
        return 0
    return first_int(text)


def normalize_location_list(value):
    if value is None:
        return set()
    if isinstance(value, list):
        text = ",".join(str(v) for v in value)
    else:
        text = str(value)
    text = text.lower().strip()
    if "no change" in text:
        return set()
    return {label for label in LOC_LABELS if label in text}


def dual_binary(dual_prior):
    road_count = int(dual_prior.get("road_count", 0) or 0)
    building_count = int(dual_prior.get("building_count", 0) or 0)
    return "yes" if road_count > 0 or building_count > 0 else "no"


def dual_count(task, dual_prior):
    if task == "count_road":
        return int(dual_prior.get("road_count", 0) or 0)
    if task == "count_build":
        return int(dual_prior.get("building_count", 0) or 0)
    return None


def dual_locations(task, dual_prior):
    if task == "loc_road":
        return normalize_location_list(dual_prior.get("road_locations", []))
    if task == "loc_build":
        return normalize_location_list(dual_prior.get("building_locations", []))
    return set()


def split_by_task(records):
    by_task = defaultdict(list)
    skipped = 0
    for row in records:
        if row.get("error"):
            skipped += 1
            continue
        if not row.get("dual_prior"):
            skipped += 1
            continue
        by_task[row.get("task", "unknown")].append(row)
    return by_task, skipped


def eval_binary(items):
    total = len(items)
    correct = 0
    invalid_ref = 0
    for row in items:
        pred = dual_binary(row["dual_prior"])
        ref = normalize_yes_no(row.get("reference", ""))
        if ref not in {"yes", "no"}:
            invalid_ref += 1
            continue
        correct += pred == ref
    valid = total - invalid_ref
    return {
        "n": total,
        "valid": valid,
        "invalid_ref": invalid_ref,
        "accuracy": correct / valid if valid else 0.0,
        "correct": correct,
    }


def eval_count(task, items):
    total = len(items)
    valid = 0
    exact = 0
    abs_error = 0
    squared_error = 0
    invalid_ref = 0
    for row in items:
        pred = dual_count(task, row["dual_prior"])
        ref = parse_reference_count(row.get("reference", ""))
        if pred is None or ref is None:
            invalid_ref += 1
            continue
        valid += 1
        err = pred - ref
        exact += pred == ref
        abs_error += abs(err)
        squared_error += err * err
    return {
        "n": total,
        "valid": valid,
        "invalid_ref": invalid_ref,
        "exact_accuracy": exact / valid if valid else 0.0,
        "mae": abs_error / valid if valid else 0.0,
        "rmse": (squared_error / valid) ** 0.5 if valid else 0.0,
    }


def eval_location(task, items):
    total = len(items)
    exact = 0
    tp = fp = fn = 0
    no_change_total = 0
    no_change_correct = 0
    for row in items:
        pred = dual_locations(task, row["dual_prior"])
        ref = normalize_location_list(row.get("reference", ""))
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


def eval_caption(items):
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge

    refs = {}
    hyps = {}
    for idx, row in enumerate(items):
        refs[idx] = [str(row.get("reference", "")).strip()]
        hyps[idx] = [str(row["dual_prior"].get("global_caption", "")).strip()]

    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
    ]

    result = {"n": len(items)}
    for scorer, names in scorers:
        score, _ = scorer.compute_score(refs, hyps)
        if isinstance(names, list):
            for name, value in zip(names, score):
                result[name] = value
        else:
            result[names] = score
    return result


def print_metric(task, metrics):
    print(f"\n[{task}]")
    for key, value in metrics.items():
        if value is None:
            print(f"{key}: None")
        elif isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Dual-Branch dual_prior fields stored in ChangeVG-Qwen prediction JSONL."
    )
    parser.add_argument("--pred-jsonl", default="/workspace/RSCD/changevg_qwen_predictions.jsonl")
    parser.add_argument("--task", default=None, help="Evaluate only one task.")
    parser.add_argument("--skip-caption", action="store_true", help="Skip caption metrics.")
    parser.add_argument("--save-json", default=None, help="Optional path to save metrics as JSON.")
    args = parser.parse_args()

    records = load_records(args.pred_jsonl)
    by_task, skipped = split_by_task(records)
    if args.task:
        by_task = {args.task: by_task.get(args.task, [])}

    all_metrics = {
        "source": args.pred_jsonl,
        "total_rows": len(records),
        "skipped_rows_without_dual_prior_or_with_error": skipped,
        "tasks": {},
    }

    print(f"Loaded rows: {len(records)}")
    print(f"Skipped rows without dual_prior or with error: {skipped}")

    for task in sorted(by_task):
        items = by_task[task]
        if not items:
            continue
        if task == "binary_class":
            metrics = eval_binary(items)
        elif task in {"count_build", "count_road"}:
            metrics = eval_count(task, items)
        elif task in {"loc_build", "loc_road"}:
            metrics = eval_location(task, items)
        elif task == "caption":
            metrics = {"n": len(items), "skipped": True} if args.skip_caption else eval_caption(items)
        else:
            metrics = {"n": len(items), "note": "No task-specific dual-prior metric implemented."}
        all_metrics["tasks"][task] = metrics
        print_metric(task, metrics)

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, ensure_ascii=False, indent=2)
        print(f"\nSaved metrics to: {args.save_json}")


if __name__ == "__main__":
    main()
