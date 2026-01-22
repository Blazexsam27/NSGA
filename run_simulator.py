"""Run ConsumptionCar.exe once with specified inputs and capture raw output.

Saves stdout/stderr to `nsga/simulation_output.txt` for inspection and parsing.
"""

import argparse
import os
import subprocess
from typing import List


def run(executable: str, inputs: List[float], timeout: int = 30):
    if not os.path.isfile(executable):
        raise FileNotFoundError(f"Executable not found: {executable}")

    cmd = [executable] + [str(x) for x in inputs]
    print("Running:", " ".join(cmd))
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception as e:
        print("Execution failed:", e)
        raise

    out_path = os.path.join(os.path.dirname(__file__), "simulation_output.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("--- STDOUT ---\n")
        f.write(res.stdout or "")
        f.write("\n--- STDERR ---\n")
        f.write(res.stderr or "")

    print(f"Saved raw output to: {out_path}")
    print("\n=== STDOUT ===\n")
    print(res.stdout)
    if res.stderr:
        print("\n=== STDERR ===\n")
        print(res.stderr)

    return out_path


def parse_args():
    p = argparse.ArgumentParser(
        description="Run ConsumptionCar.exe and capture raw output"
    )
    p.add_argument("executable", help="Path to ConsumptionCar.exe")
    p.add_argument(
        "--inputs",
        nargs=5,
        type=float,
        help="Five inputs: Iax Rtr Ig3 Ig4 Ig5",
        default=[4.0, 0.45, 2.0, 1.5, 1.0],
    )
    p.add_argument("--timeout", type=int, default=30)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        run(args.executable, args.inputs, timeout=args.timeout)
    except Exception as e:
        print("Error:", e)
        raise
