NSGA-II Design Space Exploration (Proof of Concept)

Usage

- Run the simple NSGA-II runner (uses mock simulator if `ConsumptionCar.exe` not present):

```bash
python -m nsga.nsga2
```

Files
- `simulator.py`: wrapper to call `ConsumptionCar.exe` or use a mock simulator. Also exposes a TensorFlow-batched mock evaluator.
- `nsga2.py`: NSGA-II implementation and runner. Enforces constraint `Ig3 > Ig4 > Ig5` via repair.

Notes
- The code will attempt to call `ConsumptionCar.exe` in the current working directory. If not found, it uses a deterministic mock function for objectives.
- Objectives: minimize fuel consumption (`fc`) and maximize average elasticity (`ELg3, ELg4, ELg5`) (NSGA implemented by minimizing `fc` and `-avgEL`).
 - Visualizations: The runner saves plots to an `nsga_outputs` folder in the working directory: `fc_history.png`, `pareto_front.png`, and `pareto_parameters_hist.png`.
