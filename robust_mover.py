from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from robot import MASK_DIM, SCALE, mask_to_robot, sample_mask
from simulator import Simulator
from utils import load_config


ENV_PRESETS: dict[str, dict[str, float]] = {
    "normal": {},
    "slippery": {
        "friction": 0.25,
        "drag_damping": 8.0,
    },
}

INVALID_FITNESS = -1.0


def random_genome() -> np.ndarray:
    return sample_mask(p=0.55).astype(int)


def genomes_to_robots(genomes: list[np.ndarray]) -> list[dict[str, Any]]:
    robots: list[dict[str, Any]] = []
    for genome in genomes:
        masses, springs = mask_to_robot(genome)
        masses = masses * SCALE
        robots.append(
            {
                "n_masses": int(masses.shape[0]),
                "n_springs": int(springs.shape[0]),
                "masses": masses,
                "springs": springs,
                "genome": genome.astype(int),
            }
        )
    return robots


def copy_robot(robot: dict[str, Any], include_control: bool) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in robot.items():
        if key == "control_params" and not include_control:
            continue
        if isinstance(value, np.ndarray):
            out[key] = value.copy()
        elif isinstance(value, dict):
            out[key] = {}
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, np.ndarray):
                    out[key][sub_key] = sub_value.copy()
                else:
                    out[key][sub_key] = copy.deepcopy(sub_value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def crossover(mask_a: np.ndarray, mask_b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    choose_a = rng.random((MASK_DIM, MASK_DIM)) < 0.5
    child = np.where(choose_a, mask_a, mask_b)
    return child.astype(int)


def mutate(mask: np.ndarray, mutation_rate: float, rng: np.random.Generator) -> np.ndarray:
    flips = rng.random((MASK_DIM, MASK_DIM)) < mutation_rate
    return np.where(flips, 1 - mask, mask).astype(int)


def build_sim_config(
    base_config: dict[str, Any],
    n_sims: int,
    n_masses: int,
    n_springs: int,
    env_name: str,
) -> dict[str, Any]:
    sim_config = dict(base_config["simulator"])
    sim_config.update(ENV_PRESETS[env_name])
    sim_config["n_sims"] = int(n_sims)
    sim_config["n_masses"] = int(n_masses)
    sim_config["n_springs"] = int(n_springs)
    return sim_config


def run_learning_batch(
    robots: list[dict[str, Any]],
    base_config: dict[str, Any],
    env_name: str,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, np.ndarray]], int, int]:
    max_n_masses = max(int(r["n_masses"]) for r in robots)
    max_n_springs = max(int(r["n_springs"]) for r in robots)
    sim_config = build_sim_config(
        base_config=base_config,
        n_sims=len(robots),
        n_masses=max_n_masses,
        n_springs=max_n_springs,
        env_name=env_name,
    )

    np.random.seed(seed)
    simulator = Simulator(
        sim_config=sim_config,
        taichi_config=base_config["taichi"],
        seed=seed,
        needs_grad=True,
    )
    masses = [r["masses"] for r in robots]
    springs = [r["springs"] for r in robots]
    simulator.initialize(masses, springs)

    history = simulator.train()
    final_fitness = history[:, -1]
    params = simulator.get_control_params(list(range(len(robots))))
    return final_fitness, params, max_n_masses, max_n_springs


def sanitize_fitness(values: np.ndarray, label: str) -> np.ndarray:
    arr = np.array(values, dtype=np.float32, copy=True)
    invalid_mask = ~np.isfinite(arr)
    invalid_count = int(invalid_mask.sum())
    if invalid_count > 0:
        print(
            f"  warning: replaced {invalid_count}/{arr.size} invalid fitness values "
            f"in {label} with {INVALID_FITNESS:.1f}"
        )
        arr[invalid_mask] = INVALID_FITNESS
    return arr


def evaluate_genomes_robust(
    genomes: list[np.ndarray],
    base_config: dict[str, Any],
    env_names: list[str],
    visual_env: str,
    seed_base: int,
    generation: int,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, np.ndarray]]:
    robots = genomes_to_robots(genomes)
    robust_fitness = np.zeros(len(robots), dtype=np.float32)
    per_env_fitness: dict[str, np.ndarray] = {}
    visual_params: list[dict[str, np.ndarray]] | None = None
    visual_max_n_masses: int | None = None
    visual_max_n_springs: int | None = None

    for env_idx, env_name in enumerate(env_names):
        env_seed = int(seed_base + generation * 1009 + env_idx * 97)
        fitness, control_params, max_n_masses, max_n_springs = run_learning_batch(
            robots=robots,
            base_config=base_config,
            env_name=env_name,
            seed=env_seed,
        )
        fitness = sanitize_fitness(
            fitness,
            label=f"generation={generation}, env={env_name}",
        )
        robust_fitness += fitness
        per_env_fitness[env_name] = fitness
        if env_name == visual_env:
            visual_params = control_params
            visual_max_n_masses = max_n_masses
            visual_max_n_springs = max_n_springs

    robust_fitness = sanitize_fitness(
        robust_fitness / float(len(env_names)),
        label=f"generation={generation}, aggregate",
    )
    if visual_params is None or visual_max_n_masses is None or visual_max_n_springs is None:
        raise RuntimeError(f"visual_env '{visual_env}' was not evaluated.")

    robots_with_visual_params: list[dict[str, Any]] = []
    for idx, robot in enumerate(robots):
        item = copy_robot(robot, include_control=False)
        item["control_params"] = visual_params[idx]
        item["max_n_masses"] = int(visual_max_n_masses)
        item["max_n_springs"] = int(visual_max_n_springs)
        item["trained_env"] = visual_env
        robots_with_visual_params.append(item)

    return robust_fitness, robots_with_visual_params, per_env_fitness


def evaluate_saved_robot_once(
    robot: dict[str, Any],
    base_config: dict[str, Any],
    env_name: str,
    seed: int,
) -> float:
    n_masses = int(robot.get("max_n_masses", robot["n_masses"]))
    n_springs = int(robot.get("max_n_springs", robot["n_springs"]))
    sim_config = build_sim_config(
        base_config=base_config,
        n_sims=1,
        n_masses=n_masses,
        n_springs=n_springs,
        env_name=env_name,
    )

    np.random.seed(seed)
    simulator = Simulator(
        sim_config=sim_config,
        taichi_config=base_config["taichi"],
        seed=seed,
        needs_grad=False,
    )
    simulator.initialize([robot["masses"]], [robot["springs"]])
    if "control_params" in robot:
        simulator.set_control_params([0], [robot["control_params"]])
    loss = simulator.evaluation_step()[0]
    fitness = float(-loss)
    if not np.isfinite(fitness):
        print(
            f"  warning: invalid fitness in eval env={env_name}, seed={seed}; "
            f"using {INVALID_FITNESS:.1f}"
        )
        return INVALID_FITNESS
    return fitness


def evaluate_saved_robot_multi_seed(
    robot: dict[str, Any],
    base_config: dict[str, Any],
    env_names: list[str],
    eval_seeds: list[int],
) -> dict[str, list[float]]:
    scores: dict[str, list[float]] = {}
    for env_name in env_names:
        env_scores: list[float] = []
        for seed in eval_seeds:
            env_scores.append(
                evaluate_saved_robot_once(
                    robot=robot,
                    base_config=base_config,
                    env_name=env_name,
                    seed=seed,
                )
            )
        scores[env_name] = env_scores
    return scores


def summarize_scores(scores: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for env_name, values in scores.items():
        arr = sanitize_fitness(np.array(values, dtype=np.float32), label=f"summary env={env_name}")
        summary[env_name] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
    return summary


def save_env_configs(
    output_dir: Path,
    base_config: dict[str, Any],
    env_names: list[str],
) -> None:
    for env_name in env_names:
        cfg = {
            "seed": int(base_config["seed"]),
            "taichi": dict(base_config["taichi"]),
            "simulator": dict(base_config["simulator"]),
        }
        cfg["simulator"].update(ENV_PRESETS[env_name])
        cfg["simulator"]["n_sims"] = 1
        path = output_dir / f"config_{env_name}.yaml"
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(cfg, handle, sort_keys=False)


def select_parents(
    genomes: list[np.ndarray],
    fitness: np.ndarray,
    elite_fraction: float,
) -> tuple[list[np.ndarray], np.ndarray]:
    pop_size = len(genomes)
    num_elite = max(1, int(pop_size * elite_fraction))
    ranking = np.argsort(fitness)[::-1]
    elite_idxs = ranking[:num_elite]
    parents = [genomes[i] for i in elite_idxs]
    return parents, ranking


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robust mover experiment (evolution + learning).")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--output-dir", type=str, default="robust_outputs")
    parser.add_argument("--pop-size", type=int, default=12)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--elite-fraction", type=float, default=0.25)
    parser.add_argument("--mutation-rate", type=float, default=0.03)
    parser.add_argument("--envs", type=str, default="normal,slippery")
    parser.add_argument("--visual-env", type=str, default="normal")
    parser.add_argument("--eval-seeds", type=int, default=5)
    parser.add_argument("--learning-steps", type=int, default=None)
    parser.add_argument("--sim-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = load_config(args.config)

    if args.seed is not None:
        base_config["seed"] = int(args.seed)
    if args.learning_steps is not None:
        base_config["simulator"]["learning_steps"] = int(args.learning_steps)
    if args.sim_steps is not None:
        base_config["simulator"]["sim_steps"] = int(args.sim_steps)

    env_names = [x.strip() for x in args.envs.split(",") if x.strip()]
    if not env_names:
        raise ValueError("At least one environment must be provided in --envs.")
    for env_name in env_names:
        if env_name not in ENV_PRESETS:
            raise ValueError(f"Unknown env '{env_name}'. Available: {sorted(ENV_PRESETS)}")
    if args.visual_env not in env_names:
        raise ValueError("--visual-env must be included in --envs.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Robust mover setup")
    print(f"  seed={base_config['seed']}")
    print(f"  pop_size={args.pop_size}, generations={args.generations}")
    print(f"  envs={env_names}, visual_env={args.visual_env}")
    print(
        "  sim_steps="
        f"{base_config['simulator']['sim_steps']}, learning_steps={base_config['simulator']['learning_steps']}"
    )

    np.random.seed(int(base_config["seed"]))
    rng = np.random.default_rng(int(base_config["seed"]))
    genomes = [random_genome() for _ in range(args.pop_size)]

    best_overall_robot: dict[str, Any] | None = None
    best_overall_fitness = -np.inf
    ga_history: list[dict[str, Any]] = []

    for generation in range(args.generations):
        print(f"\n=== Generation {generation} ===")
        robust_fitness, robots, per_env_fitness = evaluate_genomes_robust(
            genomes=genomes,
            base_config=base_config,
            env_names=env_names,
            visual_env=args.visual_env,
            seed_base=int(base_config["seed"]),
            generation=generation,
        )

        robust_max = float(robust_fitness.max())
        robust_mean = float(robust_fitness.mean())
        env_stats = {
            env_name: {
                "mean": float(values.mean()),
                "max": float(values.max()),
            }
            for env_name, values in per_env_fitness.items()
        }
        print(f"  robust: max={robust_max:.4f}, mean={robust_mean:.4f}")
        for env_name in env_names:
            print(
                f"  {env_name}: max={env_stats[env_name]['max']:.4f}, "
                f"mean={env_stats[env_name]['mean']:.4f}"
            )

        best_idx = int(np.argmax(robust_fitness))
        if robust_max > best_overall_fitness:
            best_overall_fitness = robust_max
            best_overall_robot = copy_robot(robots[best_idx], include_control=True)
            print(f"  new best robust fitness={best_overall_fitness:.4f}")

        ga_row: dict[str, Any] = {
            "generation": generation,
            "robust_mean": robust_mean,
            "robust_max": robust_max,
        }
        for env_name in env_names:
            ga_row[f"{env_name}_mean"] = env_stats[env_name]["mean"]
            ga_row[f"{env_name}_max"] = env_stats[env_name]["max"]
        ga_history.append(ga_row)

        parents, _ranking = select_parents(genomes, robust_fitness, args.elite_fraction)
        next_genomes = [p.copy() for p in parents]
        while len(next_genomes) < args.pop_size:
            mom = parents[int(rng.integers(len(parents)))]
            dad = parents[int(rng.integers(len(parents)))]
            child = crossover(mom, dad, rng)
            child = mutate(child, args.mutation_rate, rng)
            next_genomes.append(child)
        genomes = next_genomes[: args.pop_size]

    if best_overall_robot is None:
        raise RuntimeError("GA did not produce a best robot.")

    # Baseline morphology (before evolution), fixed and reproducible from seed.
    np.random.seed(int(base_config["seed"]) + 7_777)
    baseline_genome = random_genome()
    baseline_robot = genomes_to_robots([baseline_genome])[0]

    # Train baseline in visual env for the before_evo_after_learn mode.
    baseline_fit, baseline_params, baseline_max_masses, baseline_max_springs = run_learning_batch(
        robots=[baseline_robot],
        base_config=base_config,
        env_name=args.visual_env,
        seed=int(base_config["seed"]) + 9_001,
    )
    _ = baseline_fit

    before_evo_before = copy_robot(baseline_robot, include_control=False)
    before_evo_before["max_n_masses"] = int(baseline_max_masses)
    before_evo_before["max_n_springs"] = int(baseline_max_springs)
    before_evo_before["trained_env"] = args.visual_env

    before_evo_after = copy_robot(before_evo_before, include_control=False)
    before_evo_after["control_params"] = baseline_params[0]

    after_evo_after = copy_robot(best_overall_robot, include_control=True)
    after_evo_before = copy_robot(best_overall_robot, include_control=False)

    # Save four required modes.
    np.save(output_dir / "robust_before_evo_before_learn.npy", before_evo_before)
    np.save(output_dir / "robust_before_evo_after_learn.npy", before_evo_after)
    np.save(output_dir / "robust_after_evo_before_learn.npy", after_evo_before)
    np.save(output_dir / "robust_after_evo_after_learn.npy", after_evo_after)
    np.save(output_dir / "robust_ga_best_robot.npy", after_evo_after)

    save_env_configs(output_dir, base_config, env_names)

    eval_seed_list = [int(base_config["seed"]) + i for i in range(args.eval_seeds)]
    modes = {
        "before_evo_before_learn": before_evo_before,
        "before_evo_after_learn": before_evo_after,
        "after_evo_before_learn": after_evo_before,
        "after_evo_after_learn": after_evo_after,
    }

    mode_scores: dict[str, dict[str, list[float]]] = {}
    mode_summary: dict[str, dict[str, dict[str, float]]] = {}
    for mode_name, mode_robot in modes.items():
        scores = evaluate_saved_robot_multi_seed(
            robot=mode_robot,
            base_config=base_config,
            env_names=env_names,
            eval_seeds=eval_seed_list,
        )
        mode_scores[mode_name] = scores
        mode_summary[mode_name] = summarize_scores(scores)

    # Save machine-readable results.
    result_json = {
        "seed": int(base_config["seed"]),
        "envs": env_names,
        "visual_env": args.visual_env,
        "settings": {
            "pop_size": int(args.pop_size),
            "generations": int(args.generations),
            "elite_fraction": float(args.elite_fraction),
            "mutation_rate": float(args.mutation_rate),
            "sim_steps": int(base_config["simulator"]["sim_steps"]),
            "learning_steps": int(base_config["simulator"]["learning_steps"]),
            "eval_seeds": eval_seed_list,
        },
        "ga_history": ga_history,
        "mode_summary": mode_summary,
        "best_overall_fitness": float(best_overall_fitness),
    }
    with (output_dir / "robust_results.json").open("w", encoding="utf-8") as handle:
        json.dump(result_json, handle, indent=2)

    # Save tidy CSV for quick plotting.
    with (output_dir / "robust_eval.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mode", "env", "seed", "fitness"])
        for mode_name, env_map in mode_scores.items():
            for env_name, values in env_map.items():
                for idx, fitness in enumerate(values):
                    writer.writerow([mode_name, env_name, eval_seed_list[idx], float(fitness)])

    with (output_dir / "robust_ga_history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ga_history[0].keys()))
        writer.writeheader()
        writer.writerows(ga_history)

    print("\nSaved files:")
    print(f"  {output_dir / 'robust_before_evo_before_learn.npy'}")
    print(f"  {output_dir / 'robust_before_evo_after_learn.npy'}")
    print(f"  {output_dir / 'robust_after_evo_before_learn.npy'}")
    print(f"  {output_dir / 'robust_after_evo_after_learn.npy'}")
    print(f"  {output_dir / 'robust_results.json'}")
    print(f"  {output_dir / 'robust_eval.csv'}")
    print(f"  {output_dir / 'robust_ga_history.csv'}")

    print("\nMode summary (mean +- std):")
    for mode_name, env_map in mode_summary.items():
        print(f"  {mode_name}:")
        for env_name in env_names:
            s = env_map[env_name]
            print(f"    {env_name}: {s['mean']:.4f} +- {s['std']:.4f}")


if __name__ == "__main__":
    main()
