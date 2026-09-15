from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("outputs/g_limit")
RESULTS = Path("results/g_limit")


def accuracy(record: dict, model: str, corruption: str) -> float:
    return float(record["models"][model]["by_corruption"][corruption]["binary_accuracy"])


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    evaluation = json.loads((ROOT / "evaluation.json").read_text(encoding="utf-8"))
    replicate = json.loads((ROOT / "replicate_evaluation.json").read_text(encoding="utf-8"))
    distillation = json.loads((ROOT / "distillation_evaluation.json").read_text(encoding="utf-8"))
    sigma = json.loads((ROOT / "sigma_sensitivity.json").read_text(encoding="utf-8"))
    groups = {
        "AWGN small": ["awgn_s1", "awgn_s2", "awgn_s3"],
        "AWGN conditional": ["cond_s1", "cond_s2", "cond_s3"],
        "AWGN large": ["large_s1", "large_s2", "large_s3"],
        "Diverse small": ["diverse_s1", "diverse_s2", "diverse_s3"],
        "Diverse conditional": ["diverse_cond_s1", "diverse_cond_s2", "diverse_cond_s3"],
        "Diverse large": ["diverse_large_s1", "diverse_large_s2", "diverse_large_s3"],
    }
    key_corruptions = ["clean", "awgn_50", "awgn_100", "corr_awgn_50", "poisson_peak_10", "sinusoid_amp_30"]
    aggregate = {}
    rows = []
    for group, names in groups.items():
        aggregate[group] = {}
        for corruption in key_corruptions:
            values = np.asarray([accuracy(replicate, name, corruption) for name in names])
            aggregate[group][corruption] = {
                "mean": float(values.mean()), "std": float(values.std(ddof=1)),
                "values": values.tolist(),
            }
            rows.append(
                {"model_group": group, "corruption": corruption, "mean_accuracy": values.mean(), "std_accuracy": values.std(ddof=1)}
            )
    with (RESULTS / "replicate_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    capacity_names = ["awgn100_tiny", "awgn100", "awgn100_large", "awgn100_xlarge"]
    capacity_parameters = [evaluation["models"][name]["parameters"] for name in capacity_names]
    capacity_clean = [accuracy(evaluation, name, "clean") for name in capacity_names]
    capacity_awgn100 = [accuracy(evaluation, name, "awgn_100") for name in capacity_names]
    diverse_capacity_names = ["diverse_tiny", "diverse", "diverse_large", "diverse_xlarge"]
    diverse_corr50 = [accuracy(evaluation, name, "corr_awgn_50") for name in diverse_capacity_names]
    diverse_awgn100 = [accuracy(evaluation, name, "awgn_100") for name in diverse_capacity_names]

    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    axis = axes[0, 0]
    x = np.asarray(capacity_parameters) / 1e6
    axis.plot(x, capacity_clean, "o-", label="clean")
    axis.plot(x, capacity_awgn100, "o-", label="AWGN sigma=100")
    axis.set_xscale("log")
    axis.set_xlabel("parameters (million, log scale)")
    axis.set_ylabel("binary agreement with clean teacher")
    axis.set_title("AWGN-trained G capacity (representative runs)")
    axis.grid(alpha=0.25)
    axis.legend()

    axis = axes[0, 1]
    axis.plot(x, diverse_awgn100, "o-", label="AWGN sigma=100")
    axis.plot(x, diverse_corr50, "o-", label="correlated Gaussian sigma=50")
    axis.set_xscale("log")
    axis.set_xlabel("parameters (million, log scale)")
    axis.set_ylabel("binary agreement with clean teacher")
    axis.set_title("Diverse-noise G capacity (representative runs)")
    axis.grid(alpha=0.25)
    axis.legend()

    axis = axes[1, 0]
    labels = list(groups)
    positions = np.arange(len(labels))
    width = 0.25
    for offset, corruption in enumerate(("awgn_100", "corr_awgn_50", "sinusoid_amp_30")):
        means = [aggregate[label][corruption]["mean"] for label in labels]
        stds = [aggregate[label][corruption]["std"] for label in labels]
        axis.bar(positions + (offset - 1) * width, means, width, yerr=stds, capsize=3, label=corruption)
    axis.set_xticks(positions, labels, rotation=25, ha="right")
    axis.set_ylim(0.5, 0.9)
    axis.set_ylabel("binary agreement")
    axis.set_title("Three training seeds (mean +/- sample std)")
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.2)

    axis = axes[1, 1]
    for model_name in ("awgn50_conditional", "awgn100_conditional", "diverse_conditional_oracle", "diverse_conditional_robust"):
        records = sigma["models"][model_name]["by_true_sigma"]["100.0"]["by_supplied_sigma"]
        claims = [0, 25, 50, 75, 100, 125, 150]
        values = [records[str(claim)]["binary_accuracy"] for claim in claims]
        axis.plot(claims, values, "o-", label=model_name.replace("_conditional", " cond"))
    axis.set_xlabel("supplied sigma (true nominal AWGN sigma=100)")
    axis.set_ylabel("binary agreement")
    axis.set_title("Conditional G is sensitive to sigma calibration")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.savefig(RESULTS / "g_limit_summary.png", dpi=180)
    plt.close(figure)

    payload = {
        "protocol": evaluation["protocol"],
        "replicate_mean_std": aggregate,
        "capacity_representative_runs": {
            "parameters": capacity_parameters,
            "awgn_models": capacity_names,
            "clean_accuracy": capacity_clean,
            "awgn100_accuracy": capacity_awgn100,
            "diverse_models": diverse_capacity_names,
            "diverse_awgn100_accuracy": diverse_awgn100,
            "diverse_corr50_accuracy": diverse_corr50,
        },
        "distillation": {
            name: {
                corruption: accuracy(distillation, name, corruption)
                for corruption in key_corruptions
            }
            for name in ("diverse", "distilled", "large")
        },
    }
    (RESULTS / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))
    print(f"wrote {(RESULTS / 'g_limit_summary.png').resolve()}")


if __name__ == "__main__":
    main()
