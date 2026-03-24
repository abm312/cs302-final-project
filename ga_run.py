from argparse import ArgumentParser
import numpy as np

from utils import load_config
from simulator import Simulator
from robot import sample_mask, mask_to_robot, SCALE, MASK_DIM


def random_genome() -> np.ndarray:
    """Genome = 8x8 binary mask (same format as sample_mask)."""
    # Reuse the project’s mask sampler to get a reasonable starting shape
    return sample_mask(p=0.55)


def genomes_to_robots(genomes):
    """Convert a list of masks to robot dicts (masses + springs)."""
    robots = []
    for mask in genomes:
        masses, springs = mask_to_robot(mask)
        masses = masses * SCALE
        robots.append(
            {
                "n_masses": masses.shape[0],
                "n_springs": springs.shape[0],
                "masses": masses,
                "springs": springs,
            }
        )
    return robots


def evaluate_genomes(genomes, base_config):
    """Run the simulator on a batch of genomes and return fitness + trained robots.

    We let the simulator do its usual controller training (parallel hill climbing)
    and use the final fitness (distance travelled) as the GA fitness.
    """
    pop_size = len(genomes)

    # Deep-copy config sections we modify
    config = {
        "seed": base_config["seed"],
        "taichi": dict(base_config["taichi"]),
        "simulator": dict(base_config["simulator"]),
    }

    robots = genomes_to_robots(genomes)

    # Determine max sizes for memory allocation
    num_masses = [r["n_masses"] for r in robots]
    num_springs = [r["n_springs"] for r in robots]
    config["simulator"]["n_sims"] = pop_size
    config["simulator"]["n_masses"] = max(num_masses)
    config["simulator"]["n_springs"] = max(num_springs)

    # Initialize simulator
    simulator = Simulator(
        sim_config=config["simulator"],
        taichi_config=config["taichi"],
        seed=config["seed"],
        needs_grad=True,
    )

    masses = [r["masses"] for r in robots]
    springs = [r["springs"] for r in robots]
    simulator.initialize(masses, springs)

    # Train controllers (parallel hill climber) and take final fitness
    fitness_history = simulator.train()  # shape: (pop_size, n_learning_steps)
    fitness = fitness_history[:, -1]

    # Also grab trained control params so we can visualize the best robots later
    idxs = list(range(pop_size))
    control_params = simulator.get_control_params(idxs)
    for i, params in enumerate(control_params):
        robots[i]["control_params"] = params
        robots[i]["max_n_masses"] = config["simulator"]["n_masses"]
        robots[i]["max_n_springs"] = config["simulator"]["n_springs"]
        robots[i]["mask"] = genomes[i]

    return fitness, robots


def select_parents(genomes, fitness, elite_fraction: float):
    """Return genomes of selected parents (sorted best to worst among elites)."""
    pop_size = len(genomes)
    num_elite = max(1, int(pop_size * elite_fraction))
    ranking = np.argsort(fitness)[::-1]  # high to low
    elite_idxs = ranking[:num_elite]
    parents = [genomes[i] for i in elite_idxs]
    return parents, ranking


def crossover(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    """Uniform crossover on two masks of shape (MASK_DIM, MASK_DIM)."""
    assert mask_a.shape == mask_b.shape == (MASK_DIM, MASK_DIM)
    choose_a = np.random.rand(MASK_DIM, MASK_DIM) < 0.5
    child = np.where(choose_a, mask_a, mask_b)
    return child.astype(int)


def mutate(mask: np.ndarray, p_mut: float) -> np.ndarray:
    """Flip each cell with probability p_mut."""
    flips = np.random.rand(MASK_DIM, MASK_DIM) < p_mut
    mutated = np.where(flips, 1 - mask, mask)
    return mutated.astype(int)


def run_ga(config_path: str, pop_size: int, generations: int,
           elite_fraction: float, mutation_rate: float, seed: int | None):
    # Load base configuration (Taichi + simulator hyperparameters)
    base_config = load_config(config_path)
    if seed is not None:
        base_config["seed"] = int(seed)
    np.random.seed(base_config["seed"])

    print(f"GA config: pop_size={pop_size}, generations={generations}, "
          f"elite_fraction={elite_fraction}, mutation_rate={mutation_rate}, "
          f"seed={base_config['seed']}")

    # 1) Initial population
    genomes = [random_genome() for _ in range(pop_size)]

    best_overall_robot = None
    best_overall_fitness = -np.inf

    for gen in range(generations):
        print(f"\n=== Generation {gen} ===")
        fitness, robots = evaluate_genomes(genomes, base_config)

        max_fit = float(fitness.max())
        mean_fit = float(fitness.mean())
        print(f"  Fitness: max={max_fit:.3f}, mean={mean_fit:.3f}")

        # Track global best
        best_idx = int(np.argmax(fitness))
        if max_fit > best_overall_fitness:
            best_overall_fitness = max_fit
            best_overall_robot = robots[best_idx]
            print(f"  New best overall fitness: {best_overall_fitness:.3f}")

        # 2) Selection
        parents, ranking = select_parents(genomes, fitness, elite_fraction)

        # 3) Create next generation
        new_genomes = []
        # Elitism: copy parents directly
        new_genomes.extend([g.copy() for g in parents])

        # Fill the rest with children
        while len(new_genomes) < pop_size:
            mom, dad = np.random.choice(len(parents), size=2, replace=True)
            child = crossover(parents[mom], parents[dad])
            child = mutate(child, mutation_rate)
            new_genomes.append(child)

        genomes = new_genomes[:pop_size]

    # After last generation, save the best robot we saw
    if best_overall_robot is None:
        print("No best robot found (this should not happen).")
        return

    np.save("ga_best_robot.npy", best_overall_robot)
    print(f"\nSaved best evolved robot with fitness {best_overall_fitness:.3f} "
          f"to ga_best_robot.npy")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--pop-size", type=int, default=16,
                        help="Number of robots per generation (<= simulator.n_sims is recommended)")
    parser.add_argument("--generations", type=int, default=5,
                        help="Number of GA generations")
    parser.add_argument("--elite-fraction", type=float, default=0.25,
                        help="Fraction of population kept as parents each generation")
    parser.add_argument("--mutation-rate", type=float, default=0.02,
                        help="Per-voxel mutation probability")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (overrides config.yaml if set)")
    args = parser.parse_args()

    run_ga(
        config_path=args.config,
        pop_size=args.pop_size,
        generations=args.generations,
        elite_fraction=args.elite_fraction,
        mutation_rate=args.mutation_rate,
        seed=args.seed,
    )

