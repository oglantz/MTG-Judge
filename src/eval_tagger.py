"""
eval_tagger.py – Compare an auto-tagger's output against a manual ground-truth.

Usage
-----
    python eval_tagger.py                       # evaluates default tagger_eval.json
    python eval_tagger.py <predicted.json>      # evaluates the given file

The two JSON files must be  { "rule_id": ["tag", ...], ... }.
Every rule that appears in the ground-truth file must also appear in the
predicted file (extra rules in the predicted file are simply ignored).

Metrics reported
----------------
Per-tag   : TP / FP / FN, Precision, Recall, F1, Support (# ground-truth rules)
Per-rule  : TP / FP / FN, Precision, Recall, F1  (worst performers listed)
Aggregate : Micro-avg and Macro-avg Precision / Recall / F1
Extras    : Tag-count distribution, Jaccard similarity, coverage gap analysis
"""

import json
import sys
import os
from collections import defaultdict

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_EVAL_DIR = os.path.join(os.path.dirname(__file__), "eval")
DEFAULT_GROUND_TRUTH = os.path.join(_EVAL_DIR, "manual_tagging.json")
DEFAULT_PREDICTED    = os.path.join(_EVAL_DIR, "tagger_eval.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Return (precision, recall, F1) from raw counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return precision, recall, f1


def jaccard(set_a: set, set_b: set) -> float:
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 1.0


def bar(value: float, width: int = 20) -> str:
    filled = round(value * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def fmt_pct(v: float) -> str:
    return f"{v * 100:6.2f}%"


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate(ground_truth: dict[str, list[str]],
             predicted:    dict[str, list[str]]) -> dict:
    """
    Evaluate predicted tags against ground truth.

    Returns a dict with all intermediate and final metrics.
    """
    # Rules that appear in the ground truth
    rules = sorted(ground_truth.keys())
    missing_rules = [r for r in rules if r not in predicted]
    if missing_rules:
        print(f"WARNING: {len(missing_rules)} rules in ground truth are absent "
              f"from predicted file: {missing_rules}")

    # ---- per-rule stats ------------------------------------------------
    per_rule: dict[str, dict] = {}
    all_tags: set[str] = set()

    for rule in rules:
        gt_set   = set(ground_truth[rule])
        pred_set = set(predicted.get(rule, []))
        all_tags |= gt_set | pred_set

        tp = gt_set & pred_set
        fp = pred_set - gt_set
        fn = gt_set - pred_set

        p, r, f = prf(len(tp), len(fp), len(fn))
        per_rule[rule] = dict(
            gt=gt_set, pred=pred_set,
            tp=tp, fp=fp, fn=fn,
            precision=p, recall=r, f1=f,
            jaccard=jaccard(gt_set, pred_set),
            exact_match=(gt_set == pred_set),
        )

    # ---- per-tag stats -------------------------------------------------
    per_tag: dict[str, dict] = {}
    all_tags_sorted = sorted(all_tags)

    for tag in all_tags_sorted:
        tp = sum(1 for r in rules if tag in per_rule[r]["tp"])
        fp = sum(1 for r in rules if tag in per_rule[r]["fp"])
        fn = sum(1 for r in rules if tag in per_rule[r]["fn"])
        support = sum(1 for r in rules if tag in per_rule[r]["gt"])
        p, rec, f = prf(tp, fp, fn)
        per_tag[tag] = dict(tp=tp, fp=fp, fn=fn,
                            precision=p, recall=rec, f1=f,
                            support=support)

    # ---- aggregate (micro) --------------------------------------------
    total_tp = sum(len(v["tp"]) for v in per_rule.values())
    total_fp = sum(len(v["fp"]) for v in per_rule.values())
    total_fn = sum(len(v["fn"]) for v in per_rule.values())
    micro_p, micro_r, micro_f = prf(total_tp, total_fp, total_fn)

    # ---- aggregate (macro, per-tag mean) ------------------------------
    tags_with_support = [t for t in all_tags_sorted if per_tag[t]["support"] > 0]
    macro_p = sum(per_tag[t]["precision"] for t in tags_with_support) / len(tags_with_support)
    macro_r = sum(per_tag[t]["recall"]    for t in tags_with_support) / len(tags_with_support)
    macro_f = sum(per_tag[t]["f1"]        for t in tags_with_support) / len(tags_with_support)

    # ---- tag-count distribution ---------------------------------------
    gt_tag_counts   = [len(ground_truth[r])        for r in rules]
    pred_tag_counts = [len(predicted.get(r, []))   for r in rules]

    exact_matches = sum(1 for r in rules if per_rule[r]["exact_match"])
    avg_jaccard   = sum(per_rule[r]["jaccard"] for r in rules) / len(rules)

    return dict(
        rules=rules,
        per_rule=per_rule,
        per_tag=per_tag,
        all_tags=all_tags_sorted,
        total_tp=total_tp, total_fp=total_fp, total_fn=total_fn,
        micro_p=micro_p, micro_r=micro_r, micro_f=micro_f,
        macro_p=macro_p, macro_r=macro_r, macro_f=macro_f,
        exact_matches=exact_matches,
        avg_jaccard=avg_jaccard,
        gt_tag_counts=gt_tag_counts,
        pred_tag_counts=pred_tag_counts,
        n_rules=len(rules),
    )


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def print_report(results: dict, gt_path: str, pred_path: str) -> None:
    sep  = "=" * 72
    sep2 = "-" * 72

    print(sep)
    print("  MTG TAGGER EVALUATION REPORT")
    print(sep)
    print(f"  Ground truth : {os.path.basename(gt_path)}")
    print(f"  Predicted    : {os.path.basename(pred_path)}")
    print(f"  Rules scored : {results['n_rules']}")
    print(sep)

    # ------------------------------------------------------------------ #
    # 1. Per-tag breakdown                                                #
    # ------------------------------------------------------------------ #
    print("\n[1] PER-TAG METRICS  (only tags present in ground truth shown first)\n")

    gt_tags   = [t for t in results["all_tags"] if results["per_tag"][t]["support"] > 0]
    pred_only = [t for t in results["all_tags"] if results["per_tag"][t]["support"] == 0]

    header = f"  {'Tag':<22}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}  {'TP':>4}  {'FP':>4}  {'FN':>4}  {'Sup':>4}"
    print(header)
    print("  " + "-" * 68)

    for tag in sorted(gt_tags, key=lambda t: -results["per_tag"][t]["f1"]):
        s = results["per_tag"][tag]
        print(f"  {tag:<22}  {fmt_pct(s['precision'])}  {fmt_pct(s['recall'])}  "
              f"{fmt_pct(s['f1'])}  {s['tp']:>4}  {s['fp']:>4}  {s['fn']:>4}  {s['support']:>4}")

    if pred_only:
        print(f"\n  Tags in predicted only (hallucinated / not in ground truth):")
        for tag in sorted(pred_only):
            s = results["per_tag"][tag]
            print(f"    {tag:<22}  FP={s['fp']:>4}  (never in ground truth)")

    # ------------------------------------------------------------------ #
    # 2. Aggregate metrics                                                #
    # ------------------------------------------------------------------ #
    print(f"\n{sep2}")
    print("[2] AGGREGATE METRICS\n")

    micro_p = results["micro_p"]
    micro_r = results["micro_r"]
    micro_f = results["micro_f"]
    macro_p = results["macro_p"]
    macro_r = results["macro_r"]
    macro_f = results["macro_f"]

    print(f"  {'Metric':<18}  {'Precision':>10}  {'Recall':>10}  {'F1':>10}")
    print("  " + "-" * 52)
    print(f"  {'Micro-average':<18}  {fmt_pct(micro_p):>10}  {fmt_pct(micro_r):>10}  {fmt_pct(micro_f):>10}")
    print(f"  {'Macro-average':<18}  {fmt_pct(macro_p):>10}  {fmt_pct(macro_r):>10}  {fmt_pct(macro_f):>10}")
    print()
    print(f"  Total TP : {results['total_tp']:>5}")
    print(f"  Total FP : {results['total_fp']:>5}  (predicted but not in ground truth)")
    print(f"  Total FN : {results['total_fn']:>5}  (in ground truth but not predicted)")

    # ------------------------------------------------------------------ #
    # 3. Exact match & Jaccard                                            #
    # ------------------------------------------------------------------ #
    print(f"\n{sep2}")
    print("[3] EXACT MATCH & SIMILARITY\n")

    n = results["n_rules"]
    em = results["exact_matches"]
    print(f"  Exact-match rules   : {em}/{n}  ({em/n*100:.1f}%)")
    print(f"  Avg Jaccard / rule  : {results['avg_jaccard']:.4f}  {bar(results['avg_jaccard'])}")

    # ------------------------------------------------------------------ #
    # 4. Tag-count distribution                                           #
    # ------------------------------------------------------------------ #
    print(f"\n{sep2}")
    print("[4] TAG-COUNT DISTRIBUTION  (per rule)\n")

    gt_counts   = results["gt_tag_counts"]
    pred_counts = results["pred_tag_counts"]
    diffs       = [p - g for p, g in zip(pred_counts, gt_counts)]

    def dist(counts):
        from collections import Counter
        c = Counter(counts)
        return "  ".join(f"{k} tag(s):{v:>3}" for k, v in sorted(c.items()))

    avg_gt   = sum(gt_counts)   / n
    avg_pred = sum(pred_counts) / n

    print(f"  Ground truth avg tags/rule : {avg_gt:.2f}")
    print(f"  Predicted    avg tags/rule : {avg_pred:.2f}")
    print(f"  Avg difference (pred-gt)   : {sum(diffs)/n:+.2f}")
    print()
    print(f"  Ground truth distribution  : {dist(gt_counts)}")
    print(f"  Predicted    distribution  : {dist(pred_counts)}")

    over  = sum(1 for d in diffs if d > 0)
    under = sum(1 for d in diffs if d < 0)
    exact = sum(1 for d in diffs if d == 0)
    print(f"\n  Rules over-tagged  (pred > gt) : {over}")
    print(f"  Rules under-tagged (pred < gt) : {under}")
    print(f"  Rules same count   (pred = gt) : {exact}")

    # ------------------------------------------------------------------ #
    # 5. Worst-performing rules                                           #
    # ------------------------------------------------------------------ #
    print(f"\n{sep2}")
    print("[5] WORST-PERFORMING RULES  (lowest F1, bottom 10)\n")

    sorted_rules = sorted(results["rules"], key=lambda r: results["per_rule"][r]["f1"])
    worst = sorted_rules[:10]

    print(f"  {'Rule':<12}  {'F1':>7}  {'Prec':>7}  {'Rec':>7}  {'TP':>3}  {'FP':>3}  {'FN':>3}")
    print("  " + "-" * 56)
    for rule in worst:
        s = results["per_rule"][rule]
        print(f"  {rule:<12}  {fmt_pct(s['f1'])}  {fmt_pct(s['precision'])}  "
              f"{fmt_pct(s['recall'])}  {len(s['tp']):>3}  {len(s['fp']):>3}  {len(s['fn']):>3}")
        print(f"               GT  : {sorted(s['gt'])}")
        print(f"               Pred: {sorted(s['pred'])}")
        if s["fp"]:
            print(f"               FP  : {sorted(s['fp'])}  (extra)")
        if s["fn"]:
            print(f"               FN  : {sorted(s['fn'])}  (missed)")

    # ------------------------------------------------------------------ #
    # 6. Best-performing rules (exact match sample)                       #
    # ------------------------------------------------------------------ #
    best_exact = [r for r in results["rules"] if results["per_rule"][r]["exact_match"]]
    print(f"\n{sep2}")
    print(f"[6] EXACT-MATCH RULES  ({len(best_exact)} total)\n")
    if best_exact:
        for rule in sorted(best_exact):
            tags_str = ", ".join(sorted(results["per_rule"][rule]["gt"]))
            print(f"  {rule:<12}  {tags_str}")
    else:
        print("  None.")

    print(f"\n{sep}")
    print("  END OF REPORT")
    print(sep)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) == 2:
        pred_path = sys.argv[1]
    elif len(sys.argv) == 1:
        pred_path = DEFAULT_PREDICTED
    else:
        print("Usage: python eval_tagger.py [<predicted.json>]")
        sys.exit(1)

    gt_path = DEFAULT_GROUND_TRUTH

    for path in (gt_path, pred_path):
        if not os.path.isfile(path):
            print(f"ERROR: file not found: {path}")
            sys.exit(1)

    with open(gt_path,   encoding="utf-8") as f:
        ground_truth = json.load(f)
    with open(pred_path, encoding="utf-8") as f:
        predicted = json.load(f)

    results = evaluate(ground_truth, predicted)
    print_report(results, gt_path, pred_path)


if __name__ == "__main__":
    main()
