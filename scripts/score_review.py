"""Turn the hand-filled verdicts in reports/manual_review.csv into the one human-labelled number.

Run: uv run python -m scripts.score_review
Writes reports/manual_review_summary.md and prints the same table.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="reports/manual_review.csv")
    ap.add_argument("--out", default="reports/manual_review_summary.md")
    args = ap.parse_args()

    path = Path(args.csv)
    rows = [r for r in csv.DictReader(path.open(encoding="utf-8")) if r.get("verdict", "").strip()]
    if not rows:
        raise SystemExit(f"no verdicts filled in {path} - fill the `verdict` column first (y/n/m)")

    counts = Counter(r["verdict"].strip().lower() for r in rows)
    k = len(rows)
    y = counts["y"]
    precision = y / k

    by_driver: dict[str, Counter[str]] = {}
    for r in rows:
        by_driver.setdefault(r["driver"], Counter())[r["verdict"]] += 1

    lines = [
        "# Manual review - human-labelled precision",
        "",
        f"Reviewed **{k}** top flags from the shipped artefact "
        "(`artifacts/v0.3.0`, z-score detector).",
        "A flag counts as a true positive only if the account owner judged the",
        "transaction genuinely unusual - not merely high-scoring.",
        "",
        "| verdict | n | share |",
        "|---|---|---|",
    ]
    for label, desc in (("y", "genuinely unusual"), ("n", "normal"), ("m", "borderline")):
        lines.append(f"| {label} - {desc} | {counts[label]} | {counts[label] / k:.1%} |")
    lines += [
        "",
        f"**precision@{k} = {precision:.3f}** ({y}/{k}).",
        "",
        "## Which feature drove the flags",
        "",
        "| driver | flagged | true positive | precision |",
        "|---|---|---|---|",
    ]
    for driver, c in sorted(by_driver.items(), key=lambda kv: -sum(kv[1].values())):
        tot = sum(c.values())
        lines.append(f"| `{driver}` | {tot} | {c['y']} | {c['y'] / tot:.2f} |")

    lines += [
        "",
        "## What this says",
        "",
        "- The top of the ranking is dominated by `days_since_prev`: the detector",
        "  mostly notices long gaps between transactions, not unusual amounts.",
        "  A gap is not an anomaly - it is a holiday.",
        "- Three of the four true positives sit at 3.05, barely above the 3.0",
        "  threshold, and are also gap-driven. Only one true positive (a restaurant",
        "  charge, 5.0 sigma against its category) was amount-driven.",
        "- `days_since_prev` is clipped at 60 days, so long gaps collapse onto",
        "  identical scores, which is why the ranking contains ties.",
        "",
        "This is a finding for v0.4, not something to tune away: the feature that",
        "carries most of the signal measures the least interesting thing.",
    ]
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"reviewed {k} flags | y={counts['y']} n={counts['n']} m={counts['m']}")
    print(f"precision@{k} = {precision:.3f}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
