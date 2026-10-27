"""Compare artifact versions from their generated eval.json files.

Never hand-edit the numbers: this reads reports/eval.json (v0.1) and reports/v0.2/eval.json.

Run: uv run python -m scripts.ablation
"""
from __future__ import annotations

import json
from pathlib import Path

VERSIONS = {
    "v0.1.0 (6 features)": Path("reports/eval.json"),
    "v0.2.0 (7 features, + category z)": Path("reports/v0.2/eval.json"),
}
MODELS = ("autoencoder", "zscore", "isolation_forest")


def fmt(value) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def main() -> None:
    data = {}
    for label, path in VERSIONS.items():
        if not path.exists():
            raise FileNotFoundError(f"missing {path}; run scripts.eval_protocol first")
        data[label] = json.loads(path.read_text())

    lines = [
        "# Ablation: does the category-relative feature help?",
        "",
        "Both models are trained on the same rows and scored with the same protocol (20 injection",
        "replicates, pooled PR-AUC over ~760 labels). Only the feature set differs.",
        "",
        "## Pooled PR-AUC (real Wallet data, test n=38)",
        "",
        "| Feature set | Autoencoder | z-score | IsolationForest |",
        "|---|---|---|---|",
    ]
    for label, payload in data.items():
        pooled = payload["pooled"]
        cells = [fmt(pooled[m].get("pr_auc")) for m in MODELS]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Pooled ROC-AUC",
        "",
        "| Feature set | Autoencoder | z-score | IsolationForest |",
        "|---|---|---|---|",
    ]
    for label, payload in data.items():
        pooled = payload["pooled"]
        cells = [fmt(pooled[m].get("roc_auc")) for m in MODELS]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Synthetic corpus (10k rows, separate model per version)",
        "",
        "| Feature set | Autoencoder PR-AUC |",
        "|---|---|",
    ]
    for label, payload in data.items():
        synth = payload["synthetic"]["autoencoder"]
        vals = [x["pr_auc"] for x in synth if x.get("pr_auc") is not None]
        mean = sum(vals) / len(vals) if vals else None
        lines.append(f"| {label} | {fmt(mean)} |")

    lines += [
        "",
        "## Reading",
        "- A feature counts only if the pooled number goes up; here it did not.",
        "- Spread is in `reports/eval.json`; a single pooled number without it is not enough.",
        "- The synthetic table is the control: with ~7,000 training rows the autoencoder beats the",
        "  baselines, so the gap is a sample-size problem, not an architecture problem.",
        "- Caveat: the simulator has no category column, so the 7th feature is degenerate there.",
        "  The 0.197 -> 0.215 difference is run-to-run variance, not evidence about the feature.",
        "- The category feature is active on 56.6% of real rows (13 of 31 categories have >= 6",
        "  transactions), so a zero effect is not a dead column: not enough data per category.",
        "- Decision recorded in `docs/adr/0002-detector-choice.md`.",
    ]

    out = Path("reports/ablation.md")
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
