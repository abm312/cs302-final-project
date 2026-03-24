## Overview

This repository provides a physical simulation platform for studying automatic design of robots or [virtual creatures](https://www.nature.com/articles/s42256-019-0102-8). It is based largely on the paper, [Evolution and learning in differentiable robots](https://sites.google.com/view/eldir). By abstracting away the physical simulation and control optimization details, this codebase makes it possible to quickly iterate on algorithms for morphological design.

## Installation

1. Install [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) if you do not already have it. 
2. Create a new environment: `conda create --name alife-sim` (you can replace "alife-sim" with another name as you like).
3. Activate the environment: `conda activate alife-sim`.
4. Install Python: `conda install python=3.12`
5. Install Taichi: `pip install taichi==1.7.3`
6. Install other packages: `pip install tqdm scipy pyaml flask ipykernel matplotlib`

## Usage

1. Review the code in `run.py`. It shows an example of how to interface with the simulator.
2. Next review `config.yaml`. This includes a number of parameters, only a small number of which you should consider modifying. 
3. Review `robot.py`. This code illustrates how random robot designs can be sampled and explains the key constraints to keep in mind when representing robots for the simulator. You can also visualize designs in `visualize_robots.ipynb`. 
4. Finally, try to run the code: `python run.py`. This will generate some results files that you can visualize with `plot_fitness.ipynb` and `visualizer.py`. 

## Final Project Extension

This fork adds a "robust mover" experiment on top of the default ALife-Sim workflow.

- `ga_run.py` evolves voxel robot morphologies for locomotion.
- `robust_mover.py` evaluates bodies across two terrain presets: `normal` and `slippery`.
- `plot_robust_results.py` generates plots for the four required comparison modes.
- `robust_outputs_run2/` contains the final saved outputs used for presentation and sharing.

## What I Changed

Starting from the default ALife-Sim repository, I extended the project in a few main ways for my final project:

1. I added morphology evolution with a genetic algorithm so robot body shapes can improve across generations.
2. I added a "robust mover" experiment that evaluates robot bodies on both a normal floor and a slippery floor, instead of only using one terrain setting.
3. I organized the project around the four required comparison modes:
   before evolution / before learning,
   before evolution / after learning,
   after evolution / before learning,
   after evolution / after learning.
4. I saved the final experiment outputs, plots, and robot files so the results can be reproduced and visualized directly from this repository.
5. I also made small visualization and robot-generation updates to support the final demos.

## Where The Edited Code Is

If you want to inspect the code that was added or changed for this project, start with these files:

- `ga_run.py`: added to evolve voxel robot morphologies with a genetic algorithm.
- `robust_mover.py`: added for the full final-project pipeline, including robust evaluation, four-mode export, and saved results.
- `plot_robust_results.py`: added to generate the bar chart and GA-history plots from experiment outputs.
- `robot.py`: edited to support additional robot sampling options used during experimentation.
- `run.py`: edited to support choosing robot types from the command line.
- `visualizer.py`: edited so saved robots from the new experiments can be visualized correctly.
- `visualizer/templates/index.html` and `visualizer/static/style.css`: edited to improve the project visualizer.
- `plot_fitness.ipynb`: edited for plotting and inspecting learning results.
- `robust_outputs_run2/`: final experiment outputs used in the shared public version of the project.

The four required modes are:

1. Before evolution, before learning
2. Before evolution, after learning
3. After evolution, before learning
4. After evolution, after learning

To reproduce the final robust mover experiment:

```bash
python robust_mover.py \
  --config config.yaml \
  --output-dir robust_outputs_final \
  --pop-size 10 \
  --generations 4 \
  --envs normal,slippery \
  --visual-env normal \
  --eval-seeds 5 \
  --learning-steps 20 \
  --sim-steps 700 \
  --seed 7
```

To generate plots from an output folder:

```bash
python plot_robust_results.py --input-dir robust_outputs_run2
```

To visualize one saved robot:

```bash
python visualizer.py \
  --input robust_outputs_run2/robust_after_evo_after_learn.npy \
  --config robust_outputs_run2/config_normal.yaml
```
