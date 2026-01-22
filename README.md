NSGA-II Design Space Exploration (Proof of Concept)

Usage

Files
- `simulator.py`: wrapper to call `ConsumptionCar.exe` or use a mock simulator. Also exposes a TensorFlow-batched mock evaluator.
- `nsga2.py`: NSGA-II implementation and runner. Enforces constraint `Ig3 > Ig4 > Ig5` via repair.

Notes
- The code will attempt to call `ConsumptionCar.exe` in the current working directory. If not found, it uses a deterministic mock function for objectives.
- Objectives: minimize fuel consumption (`fc`) and maximize average elasticity (`ELg3, ELg4, ELg5`) (NSGA implemented by minimizing `fc` and `-avgEL`).
 - Visualizations: The runner saves plots to an `nsga_outputs` folder in the working directory: `fc_history.png`, `pareto_front.png`, and `pareto_parameters_hist.png`.
 - 
# NSGA-II Design Space Exploration (Proof of Concept)

This module contains a small proof-of-concept NSGA-II Design Space Exploration runner that evaluates a (mock or real) `ConsumptionCar.exe` simulator over five inputs and produces a Pareto front and visualizations.

## Quick setup (Windows PowerShell)

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies (from project root):

```powershell
pip install -r requirements.txt
```

## Run

From the project root (recommended):

```powershell
python -m nsga.nsga2
```

Or run directly from the `nsga` folder:

```powershell
cd nsga
python nsga2.py
```

Strict simulator usage

This project can be run in two modes:

- permissive (default): if `ConsumptionCar.exe` is missing or its output cannot be parsed, the runner falls back to an internal mock simulator (useful for development and demos).
- strict: the runner will require a working `ConsumptionCar.exe` and will raise an error if the executable is missing, fails to run, or returns unparsable output.

To run with the real simulator in strict mode (recommended for your guideline), place `ConsumptionCar.exe` in the project root or pass an explicit path and set `strict_simulator=True` when calling `run_nsga` programmatically:

```python
from nsga import nsga2
nsga2.run_nsga(pop_size=40, generations=25, executable_path=r"C:\path\to\ConsumptionCar.exe", strict_simulator=True)
```

If you prefer to run the script from the `nsga` folder and enforce strict mode, call the script via a short wrapper or edit the `if __name__ == "__main__"` section to pass `strict_simulator=True` (I can add a CLI flag if you want).

### Capturing raw simulator output

If you don't know the `ConsumptionCar.exe` stdout format, run the included helper to capture one raw run for inspection:

From the `nsga` folder:

```powershell
python run_simulator.py "C:\full\path\to\ConsumptionCar.exe" --inputs 4.0 0.45 2.0 1.5 1.0
```

This will write `nsga/simulation_output.txt` containing the full stdout and stderr. Paste that file's contents here (or attach) and I will update `_parse_output` to correctly extract the four outputs.

## Outputs
All outputs are saved to the `nsga` module directory. Key files produced:

- `pareto_front.csv` — Pareto solutions with columns: `Iax, Rtr, Ig3, Ig4, Ig5, fc, neg_avgEL` (`neg_avgEL` is negative average elasticity used as minimized objective).
- `fc_history.png` — best & mean fuel consumption per generation.
- `pareto_front.png` — scatter of Pareto solutions in objective space (fc vs avgEL).
- `pareto_parameters_hist.png` — histograms of parameter values for Pareto solutions.
- `population_parameters_hist.png` — histograms for the entire final population (useful when Pareto is small).
- `population_objectives_fronts.png` — scatter of all individuals colored by Pareto front rank.

## What each graph tells you

- `fc_history.png` (Objective history): shows algorithm convergence over generations. A downward trend in "Best fc" indicates improvement in fuel consumption. The gap between best and mean shows population diversity.

- `pareto_front.png` (Pareto front): visualizes the trade-off between fuel consumption (`fc`, to minimize) and average elasticity (`avgEL`, to maximize). Points on the left/top are better on `fc` and `avgEL` respectively. Because the code minimizes `-avgEL`, the plotted y-axis shows the true `avgEL`.

- `pareto_parameters_hist.png` (Pareto parameter histograms): shows the distribution of input variables among Pareto solutions — useful to see which design variables concentrate on the best trade-offs.

- `population_parameters_hist.png` (Full-pop histograms): shows the parameter distribution across the whole final population. Use this to compare how Pareto solutions differ from the population at large.

- `population_objectives_fronts.png` (Population scatter & fronts): plots all evaluated individuals in objective space and colors them by Pareto front rank (`Front 0` = nondominated front / Pareto front, `Front 1` = dominated by front 0, etc.). Lower-numbered fronts are strictly better (not dominated by lower-numbered fronts). This helps you inspect how many distinct Pareto fronts exist and how crowded the objective space is.


