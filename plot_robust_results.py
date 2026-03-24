from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot robust mover experiment outputs.")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def load_eval_csv(path: Path) -> dict[str, dict[str, list[float]]]:
    data: dict[str, dict[str, list[float]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            mode = row["mode"]
            env = row["env"]
            val = float(row["fitness"])
            data.setdefault(mode, {}).setdefault(env, []).append(val)
    return data


def load_ga_history(path: Path) -> tuple[list[int], dict[str, list[float]]]:
    generations: list[int] = []
    series: dict[str, list[float]] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            generations.append(int(row["generation"]))
            for key, value in row.items():
                if key == "generation":
                    continue
                series.setdefault(key, []).append(float(value))
    return generations, series


def plot_mode_bars(eval_data: dict[str, dict[str, list[float]]], output_path: Path) -> None:
    mode_order = [
        "before_evo_before_learn",
        "before_evo_after_learn",
        "after_evo_before_learn",
        "after_evo_after_learn",
    ]
    envs = sorted({env for mode_map in eval_data.values() for env in mode_map.keys()})
    x = np.arange(len(mode_order), dtype=np.float32)
    width = 0.38 if len(envs) <= 2 else max(0.15, 0.8 / len(envs))

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

    for idx, env in enumerate(envs):
        means = []
        stds = []
        for mode in mode_order:
            values = np.array(eval_data.get(mode, {}).get(env, [0.0]), dtype=np.float32)
            means.append(float(values.mean()))
            stds.append(float(values.std()))
        shift = (idx - (len(envs) - 1) / 2.0) * width
        ax.bar(x + shift, means, width=width, label=env, alpha=0.85)
        ax.errorbar(x + shift, means, yerr=stds, fmt="none", ecolor="black", capsize=3, linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            "Before Evo\nBefore Learn",
            "Before Evo\nAfter Learn",
            "After Evo\nBefore Learn",
            "After Evo\nAfter Learn",
        ]
    )
    ax.set_ylabel("Fitness (Forward Distance)")
    ax.set_title("Robust Mover: 4-Mode Fitness by Environment")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_ga_history(generations: list[int], series: dict[str, list[float]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

    # Prefer mean series for readability
    for key in sorted(series.keys()):
        if key.endswith("_mean"):
            ax.plot(generations, series[key], marker="o", linewidth=2, label=key)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title("Robust Mover GA History")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_path = input_dir / "robust_eval.csv"
    ga_path = input_dir / "robust_ga_history.csv"
    if not eval_path.exists():
        raise FileNotFoundError(f"Missing {eval_path}")
    if not ga_path.exists():
        raise FileNotFoundError(f"Missing {ga_path}")

    eval_data = load_eval_csv(eval_path)
    generations, series = load_ga_history(ga_path)

    bars_path = output_dir / "robust_mode_bars.png"
    ga_curve_path = output_dir / "robust_ga_history.png"
    plot_mode_bars(eval_data, bars_path)
    plot_ga_history(generations, series, ga_curve_path)

    print("Saved plots:")
    print(f"  {bars_path}")
    print(f"  {ga_curve_path}")


if __name__ == "__main__":
    main()
