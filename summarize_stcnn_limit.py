from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    output = Path("results/stcnn_limit")
    output.mkdir(parents=True, exist_ok=True)
    core = load("outputs/stcnn_limit/core/summary.json")
    high = load("outputs/stcnn_limit/train_sigma100/summary.json")
    high_distilled = load("outputs/stcnn_limit/distillation_sigma100/summary.json")
    corruptions = load("outputs/stcnn_limit/corruptions_after_diverse.json")
    diverse_distilled = load("outputs/stcnn_limit/diverse_distillation_eval.json")
    robust_corruptions = load("outputs/stcnn_limit/sigma_robust_corruptions.json")
    multi_seed = load("outputs/stcnn_limit/diverse_multiseed_eval.json")
    sigma_sensitivity = load("outputs/stcnn_limit/sigma_robust_sensitivity.json")
    synthetic = load("outputs/stcnn_limit/sigma_robust_synthetic_signal.json")
    full_image = load("results/stcnn_limit/diverse_distilled_demo_awgn.json")

    awgn_models = {
        "small, train <=50": core["models"]["student_paired"],
        "small + sigma, train <=50": core["models"]["student_paired_conditional"],
        "small, train <=100": high["models"]["student_paired"],
        "large teacher, train <=100": high["models"]["teacher_large_standard"],
        "small distilled, train <=100": high_distilled["models"]["student_paired_distilled"],
    }
    sigmas = [0.0, 5.0, 15.0, 25.0, 35.0, 50.0, 60.0, 75.0, 100.0]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    for name, record in awgn_models.items():
        axes[0].plot(
            sigmas,
            [100.0 * record["by_sigma"][str(sigma)]["accuracy"] for sigma in sigmas],
            marker="o",
            linewidth=2,
            label=name,
        )
    axes[0].axvline(50, color="gray", linestyle="--", linewidth=1)
    axes[0].set_title("Held-out patch accuracy vs AWGN")
    axes[0].set_xlabel("AWGN sigma on 8-bit scale")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(75, 100)
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    selected_corruptions = [
        "clean",
        "awgn_100",
        "corr_awgn_50",
        "poisson_peak_10",
        "saltpepper_prob_0.05",
        "sinusoid_amp_30",
    ]
    corruption_models = {
        "AWGN100 small": corruptions["models"]["paired100"]["by_corruption"],
        "diverse distilled": diverse_distilled["models"]["diverse_distilled"]["by_corruption"],
        "diverse + exact sigma": corruptions["models"]["diverse_conditioned"]["by_corruption"],
        "diverse large teacher": corruptions["models"]["diverse_teacher"]["by_corruption"],
        "diverse + sigma jitter": robust_corruptions["models"]["robust"]["by_corruption"],
    }
    x = np.arange(len(selected_corruptions))
    width = 0.16
    for index, (name, record) in enumerate(corruption_models.items()):
        axes[1].bar(
            x + (index - 2) * width,
            [100.0 * record[item]["accuracy"] for item in selected_corruptions],
            width,
            label=name,
        )
    axes[1].set_title("Unseen/family-shift corruption accuracy")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim(45, 100)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        ["clean", "AWGN 100", "corr. G 50", "Poisson 10", "S&P 5%", "sine amp 30"],
        rotation=20,
        ha="right",
    )
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "stcnn_limit_summary.png", dpi=170)
    plt.close(fig)

    multi_seed_summary = {}
    for prefix in ("cond", "teacher"):
        names = [f"{prefix}_seed{index}" for index in (1, 2, 3)]
        multi_seed_summary[prefix] = {}
        for corruption in ("clean", "awgn_50", "awgn_100", "corr_awgn_50", "sinusoid_amp_30"):
            values = np.array([
                multi_seed["models"][name]["by_corruption"][corruption]["accuracy"]
                for name in names
            ])
            multi_seed_summary[prefix][corruption] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
                "values": values.tolist(),
            }

    full_image_stability = {}
    for sigma in (15.0, 50.0, 100.0):
        records = [item for item in full_image["noise_stability"] if item["sigma"] == sigma]
        full_image_stability[str(sigma)] = {
            "mean_binary_agreement_with_clean": float(
                np.mean([item["binary_agreement_with_sigma0"] for item in records])
            ),
            "mean_probability_correlation_with_clean": float(
                np.mean([item["probability_correlation_with_sigma0"] for item in records])
            ),
            "mean_absolute_difference_from_clean": float(
                np.mean([item["mean_absolute_difference_from_sigma0"] for item in records])
            ),
        }

    payload = {
        "protocol": {
            "natural_train_patches": 6000,
            "natural_validation_patches": 2000,
            "validation_noise_repeats": 5,
            "labels": "clean-image Sobel-extreme hard pseudo-labels",
            "important_scope_warning": "Patch accuracy is not human texture GT or dense full-image accuracy.",
        },
        "awgn_key_accuracy": {
            name: {
                sigma: record["by_sigma"][sigma]["accuracy"]
                for sigma in ("0.0", "50.0", "75.0", "100.0")
            }
            for name, record in awgn_models.items()
        },
        "diverse_corruption_key_accuracy": {
            name: {
                corruption: record[corruption]["accuracy"]
                for corruption in selected_corruptions
            }
            for name, record in corruption_models.items()
        },
        "diverse_training_three_seed_reproduction": multi_seed_summary,
        "sigma_input_true_awgn50": {
            name: {
                claim: sigma_sensitivity["models"][name]["by_true_sigma"]["50.0"][claim]["accuracy"]
                for claim in ("fixed_0", "fixed_25", "fixed_50", "fixed_75", "fixed_100", "oracle_effective_rms")
            }
            for name in ("exact", "robust")
        },
        "synthetic_signal": {
            name: {
                corruption: synthetic["models"][name]["by_corruption"][corruption][
                    "predicted_texture_rate_by_clean_signal"
                ]
                for corruption in ("clean", "awgn_50", "corr_awgn_50")
            }
            for name in ("exact", "robust", "nonconditional")
        },
        "distilled_full_image_awgn_stability": full_image_stability,
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rows = []
    for name, record in awgn_models.items():
        for sigma in sigmas:
            rows.append(
                {
                    "section": "awgn_curve",
                    "model": name,
                    "condition": f"sigma_{sigma:g}",
                    "accuracy": record["by_sigma"][str(sigma)]["accuracy"],
                }
            )
    for name, record in corruption_models.items():
        for corruption in selected_corruptions:
            rows.append(
                {
                    "section": "corruption",
                    "model": name,
                    "condition": corruption,
                    "accuracy": record[corruption]["accuracy"],
                }
            )
    with (output / "key_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {(output / 'summary.json').resolve()}")
    print(f"wrote {(output / 'key_metrics.csv').resolve()}")
    print(f"wrote {(output / 'stcnn_limit_summary.png').resolve()}")


if __name__ == "__main__":
    main()
