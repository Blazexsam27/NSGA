import math
import random
import numpy as np
import csv
import os
from typing import List, Tuple

try:
    from nsga.simulator import Simulator
except Exception:
    # allow running the script directly from the `nsga` folder
    from simulator import Simulator
import tensorflow as tf
import matplotlib.pyplot as plt


def ensure_bounds(x, bounds):
    x = np.array(x, dtype=float)
    for i, (lo, hi) in enumerate(bounds):
        x[i] = float(np.clip(x[i], lo, hi))
    return x


def repair_gears(x, gear_idxs, gear_bounds):
    # Enforce ig3 > ig4 > ig5 by sorting the gear values descending and clipping
    gears = sorted([x[i] for i in gear_idxs], reverse=True)
    for idx, val in zip(gear_idxs, gears):
        lo, hi = gear_bounds
        x[idx] = float(np.clip(val, lo, hi))
    return x


def initialize_population(pop_size, bounds, gear_idxs, gear_bounds):
    pop = []
    for _ in range(pop_size):
        # sample Iax and Rtr independently
        Iax = random.uniform(bounds[0][0], bounds[0][1])
        Rtr = random.uniform(bounds[1][0], bounds[1][1])

        # sample gears ensuring ig3>ig4>ig5
        g5 = random.uniform(gear_bounds[0], gear_bounds[1])
        g4 = random.uniform(g5, gear_bounds[1])
        g3 = random.uniform(g4, gear_bounds[1])

        ind = np.array([Iax, Rtr, g3, g4, g5], dtype=float)
        pop.append(ind)
    return pop


def dominates(a, b):
    # a and b are objective vectors (to minimize)
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def fast_nondominated_sort(objs: List[Tuple[float, float]]):
    S = [[] for _ in range(len(objs))]
    n = [0] * len(objs)
    rank = [0] * len(objs)
    fronts = [[]]

    for p in range(len(objs)):
        for q in range(len(objs)):
            if dominates(objs[p], objs[q]):
                S[p].append(q)
            elif dominates(objs[q], objs[p]):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        Q = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    Q.append(q)
        i += 1
        fronts.append(Q)
    fronts.pop()  # last is empty
    return fronts


def crowding_distance(objs, front):
    distance = [0.0] * len(front)
    if len(front) == 0:
        return distance
    num_obj = len(objs[0])
    for m in range(num_obj):
        values = [(objs[i][m], idx) for idx, i in enumerate(front)]
        values.sort()
        f_min = values[0][0]
        f_max = values[-1][0]
        if f_max - f_min == 0:
            continue
        distance[0] = float("inf")
        distance[-1] = float("inf")
        for k in range(1, len(values) - 1):
            prev_val = values[k - 1][0]
            next_val = values[k + 1][0]
            distance[k] += (next_val - prev_val) / (f_max - f_min)
    return distance


def tournament_selection(pop, objs, k=2):
    i, j = random.sample(range(len(pop)), 2)
    if dominates(objs[i], objs[j]):
        return pop[i]
    if dominates(objs[j], objs[i]):
        return pop[j]
    return pop[i] if random.random() < 0.5 else pop[j]


def simulated_binary_crossover(a, b, eta=15.0, bounds=None):
    a = np.copy(a)
    b = np.copy(b)
    child1 = np.copy(a)
    child2 = np.copy(b)
    for i in range(len(a)):
        if random.random() <= 0.5:
            if abs(a[i] - b[i]) > 1e-14:
                x1 = min(a[i], b[i])
                x2 = max(a[i], b[i])
                rand = random.random()
                beta = 1.0 + (2.0 * (x1 - bounds[i][0]) / (x2 - x1))
                alpha = 2.0 - beta ** -(eta + 1)
                if rand <= 1.0 / alpha:
                    betaq = (rand * alpha) ** (1.0 / (eta + 1))
                else:
                    betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
                c1 = 0.5 * ((x1 + x2) - betaq * (x2 - x1))

                beta = 1.0 + (2.0 * (bounds[i][1] - x2) / (x2 - x1))
                alpha = 2.0 - beta ** -(eta + 1)
                if rand <= 1.0 / alpha:
                    betaq = (rand * alpha) ** (1.0 / (eta + 1))
                else:
                    betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
                c2 = 0.5 * ((x1 + x2) + betaq * (x2 - x1))
                child1[i] = float(np.clip(c1, bounds[i][0], bounds[i][1]))
                child2[i] = float(np.clip(c2, bounds[i][0], bounds[i][1]))
            else:
                child1[i] = a[i]
                child2[i] = b[i]
        else:
            child1[i] = a[i]
            child2[i] = b[i]
    return child1, child2


def polynomial_mutation(x, eta=20.0, p=0.1, bounds=None):
    y = np.copy(x)
    for i in range(len(x)):
        if random.random() < p:
            u = random.random()
            lo, hi = bounds[i]
            delta1 = (y[i] - lo) / (hi - lo)
            delta2 = (hi - y[i]) / (hi - lo)
            mut_pow = 1.0 / (eta + 1.0)
            if u < 0.5:
                xy = 1.0 - delta1
                val = 2.0 * u + (1.0 - 2.0 * u) * (xy ** (eta + 1))
                deltaq = val**mut_pow - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (xy ** (eta + 1))
                deltaq = 1.0 - val**mut_pow
            y[i] = float(np.clip(y[i] + deltaq * (hi - lo), lo, hi))
    return y


def evaluate_population(pop, simulator: Simulator):
    X = np.vstack(pop)
    # Use TF batch evaluator when using mock for speed
    if simulator.use_mock:
        fc, ELg3, ELg4, ELg5 = simulator.evaluate_batch_tf(X)
        objs = []
        for f, e3, e4, e5 in zip(fc, ELg3, ELg4, ELg5):
            # minimize fc and minimize negative average elasticity (i.e., maximize elasticity)
            objs.append((float(f), float(-(e3 + e4 + e5) / 3.0)))
        return objs

    objs = []
    for x in pop:
        r = simulator.evaluate(x)
        f = r["fc"]
        avgEL = (r["ELg3"] + r["ELg4"] + r["ELg5"]) / 3.0
        objs.append((float(f), float(-avgEL)))
    return objs


def run_nsga(
    pop_size=50,
    generations=30,
    crossover_prob=0.9,
    mutation_prob=0.2,
    seed=None,
    executable_path=None,
    strict_simulator=False,
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    # Variable order: [Iax, Rtr, Ig3, Ig4, Ig5]
    bounds = [(3.00, 5.50), (0.40, 0.50), (0.50, 2.25), (0.50, 2.25), (0.50, 2.25)]
    gear_idxs = [2, 3, 4]
    gear_bounds = (0.50, 2.25)

    simulator = Simulator(executable_path, strict=strict_simulator)

    pop = initialize_population(pop_size, bounds, gear_idxs, gear_bounds)

    # outputs directory: place outputs in the same directory as this script
    file_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = file_dir
    os.makedirs(out_dir, exist_ok=True)

    gen_best_fc = []
    gen_mean_fc = []
    gen_pareto_sizes = []

    for gen in range(generations):
        objs = evaluate_population(pop, simulator)

        # create offspring population
        offspring = []
        while len(offspring) < pop_size:
            parent1 = tournament_selection(pop, objs)
            parent2 = tournament_selection(pop, objs)
            if random.random() < crossover_prob:
                c1, c2 = simulated_binary_crossover(parent1, parent2, bounds=bounds)
            else:
                c1, c2 = parent1.copy(), parent2.copy()
            c1 = polynomial_mutation(c1, p=mutation_prob, bounds=bounds)
            c2 = polynomial_mutation(c2, p=mutation_prob, bounds=bounds)
            # repair gear ordering
            c1 = repair_gears(c1, gear_idxs, gear_bounds)
            c2 = repair_gears(c2, gear_idxs, gear_bounds)
            offspring.append(c1)
            if len(offspring) < pop_size:
                offspring.append(c2)

        # combine and select next generation
        combined = pop + offspring
        combined_objs = evaluate_population(combined, simulator)
        fronts = fast_nondominated_sort(combined_objs)

        new_pop = []
        for front in fronts:
            if len(new_pop) + len(front) > pop_size:
                # crowding distance sort
                front_objs = [combined_objs[i] for i in front]
                distances = crowding_distance(combined_objs, front)
                # pair indices with distances
                paired = list(zip(front, distances))
                paired.sort(key=lambda t: t[1], reverse=True)
                for idx, _ in paired:
                    if len(new_pop) < pop_size:
                        new_pop.append(combined[idx])
                break
            else:
                for idx in front:
                    new_pop.append(combined[idx])
        pop = new_pop

        # logging
        best_obj = min(combined_objs, key=lambda o: o[0])
        print(
            f"Gen {gen+1}/{generations} | Best fc: {best_obj[0]:.4f}, obj2: {best_obj[1]:.4f}"
        )
        # record metrics
        f_vals = [o[0] for o in combined_objs]
        gen_best_fc.append(float(min(f_vals)))
        gen_mean_fc.append(float(sum(f_vals) / len(f_vals)))
        # approximate current Pareto size
        fronts_now = fast_nondominated_sort(combined_objs)
        gen_pareto_sizes.append(len(fronts_now[0]) if fronts_now else 0)

    # final Pareto front
    final_objs = evaluate_population(pop, simulator)
    fronts = fast_nondominated_sort(final_objs)
    pareto_idx = fronts[0]
    pareto = [(pop[i], final_objs[i]) for i in pareto_idx]

    # save Pareto front (in module directory)
    with open(os.path.join(out_dir, "pareto_front.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Iax", "Rtr", "Ig3", "Ig4", "Ig5", "fc", "neg_avgEL"])
        for x, obj in pareto:
            writer.writerow(list(x) + list(obj))

    print(f"Saved Pareto front with {len(pareto)} solutions to pareto_front.csv")

    # --- Visualizations ---
    # 1) Objective history
    try:
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(gen_best_fc) + 1), gen_best_fc, label="Best fc")
        plt.plot(range(1, len(gen_mean_fc) + 1), gen_mean_fc, label="Mean fc")
        plt.xlabel("Generation")
        plt.ylabel("Fuel Consumption (fc)")
        plt.title("Objective over generations")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "fc_history.png"))
        plt.close()
    except Exception:
        pass

    # 2) Pareto scatter (fc vs avgEL)
    try:
        plt.figure(figsize=(6, 6))
        xs = [obj[0] for _, obj in pareto]
        ys = [-obj[1] for _, obj in pareto]  # avgEL
        plt.scatter(xs, ys, c="tab:blue")
        plt.xlabel("Fuel Consumption (fc)")
        plt.ylabel("Average Elasticity (avgEL)")
        plt.title("Pareto Front (fc vs avgEL)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "pareto_front.png"))
        plt.close()
    except Exception:
        pass

    # 3) Parameter histograms for Pareto solutions
    try:
        if pareto:
            params = np.vstack([x for x, _ in pareto])
            names = ["Iax", "Rtr", "Ig3", "Ig4", "Ig5"]
            plt.figure(figsize=(10, 6))
            for i in range(params.shape[1]):
                plt.subplot(2, 3, i + 1)
                plt.hist(params[:, i], bins=8, color="C%d" % i)
                plt.title(names[i])
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "pareto_parameters_hist.png"))
            plt.close()
    except Exception:
        pass

    # 4) Histograms for the entire final population (helpful when Pareto is small)
    try:
        params_all = np.vstack(pop)
        names = ["Iax", "Rtr", "Ig3", "Ig4", "Ig5"]
        plt.figure(figsize=(10, 6))
        for i in range(params_all.shape[1]):
            plt.subplot(2, 3, i + 1)
            plt.hist(params_all[:, i], bins=10, color="C%d" % i)
            plt.title(names[i])
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "population_parameters_hist.png"))
        plt.close()
    except Exception:
        pass

    # 5) Scatter of all individuals in objective space colored by Pareto front rank
    try:
        xs_all = [o[0] for o in final_objs]
        ys_all = [-o[1] for o in final_objs]
        # compute ranks per solution
        rank_of = {}
        for r, front in enumerate(fronts):
            for idx in front:
                rank_of[idx] = r
        ranks = [rank_of.get(i, 999) for i in range(len(final_objs))]
        cmap = plt.get_cmap("tab10")
        plt.figure(figsize=(6, 6))
        for r in sorted(set(ranks)):
            idxs = [i for i, rr in enumerate(ranks) if rr == r]
            if not idxs:
                continue
            plt.scatter(
                [xs_all[i] for i in idxs],
                [ys_all[i] for i in idxs],
                label=f"Front {r}",
                color=cmap(r % 10),
                s=30,
            )
        plt.xlabel("Fuel Consumption (fc)")
        plt.ylabel("Average Elasticity (avgEL)")
        plt.title("Population objectives colored by front")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "population_objectives_fronts.png"))
        plt.close()
    except Exception:
        pass

    print(f"Saved visualizations to {out_dir}")
    return pareto


if __name__ == "__main__":
    run_nsga(pop_size=40, generations=20, seed=0)
