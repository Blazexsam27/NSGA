import os
import subprocess
import numpy as np
import tensorflow as tf


class Simulator:
    """Wrapper to call ConsumptionCar.exe or use a mock simulator when not available.

    The interface expects an input vector [Iax, Rtr, Ig3, Ig4, Ig5].
    It returns a dict with keys: `fc`, `ELg3`, `ELg4`, `ELg5`.
    """

    def __init__(self, executable_path=None, strict=False):
        if executable_path is None:
            executable_path = os.path.join(os.getcwd(), "ConsumptionCar.exe")
        self.executable_path = executable_path
        self.strict = bool(strict)
        self.use_mock = not os.path.isfile(self.executable_path)
        if self.strict and self.use_mock:
            raise FileNotFoundError(
                f"ConsumptionCar executable not found at: {self.executable_path} (strict mode)"
            )

    def evaluate(self, x):
        """Evaluate a single input vector or sequence.

        x: array-like with shape (5,) -> [Iax, Rtr, Ig3, Ig4, Ig5]
        """
        x = np.asarray(x, dtype=float)
        if x.ndim != 1 or x.size != 5:
            raise ValueError("Input must be a 1D array of length 5")

        if self.use_mock:
            if self.strict:
                raise FileNotFoundError(
                    f"ConsumptionCar executable not found at: {self.executable_path} (strict mode)"
                )
            return self._mock_eval(x)

        # Real executable call (best-effort, depends on ConsumptionCar interface)
        # We'll pass values as command-line args and expect stdout with lines like 'fc: <val>' etc.
        cmd = [self.executable_path] + [f"{v:.6g}" for v in x.tolist()]
        try:
            out = subprocess.run(
                cmd, capture_output=True, text=True, check=True, timeout=20
            )
            parsed = self._parse_output(out.stdout)
            # If parsing failed, handle according to strict flag
            if any(v is None for v in parsed.values()):
                if self.strict:
                    raise ValueError(
                        f"ConsumptionCar output could not be parsed (strict mode). Raw output:\n{out.stdout}"
                    )
                print(
                    "Warning: simulator output could not be parsed; switching to mock simulator."
                )
                self.use_mock = True
                return self._mock_eval(x)
            return parsed
        except Exception as exc:
            if self.strict:
                raise RuntimeError(
                    f"Execution of ConsumptionCar failed (strict mode): {exc}"
                )
            # fallback to mock if any issue
            print("Warning: simulator execution failed; using mock simulator.")
            self.use_mock = True
            return self._mock_eval(x)

    def evaluate_batch_tf(self, X):
        """Evaluate a batch using TensorFlow ops (mock only). Returns tuple of arrays: fc, ELg3, ELg4, ELg5"""
        X = tf.convert_to_tensor(X, dtype=tf.float32)

        # X shape: (N,5)
        Iax = X[:, 0]
        Rtr = X[:, 1]
        Ig3 = X[:, 2]
        Ig4 = X[:, 3]
        Ig5 = X[:, 4]

        # Mock physics-inspired formulas
        fc = 1000.0 / (Iax * Rtr * (Ig3 + Ig4 + Ig5) + 1e-6)
        ELg3 = 0.1 * Ig3 + 0.02 * Iax
        ELg4 = 0.1 * Ig4 + 0.02 * Iax
        ELg5 = 0.1 * Ig5 + 0.02 * Iax

        return fc.numpy(), ELg3.numpy(), ELg4.numpy(), ELg5.numpy()

    def _mock_eval(self, x):
        Iax, Rtr, Ig3, Ig4, Ig5 = x.tolist()
        fc = 1000.0 / (Iax * Rtr * (Ig3 + Ig4 + Ig5) + 1e-6)
        ELg3 = 0.1 * Ig3 + 0.02 * Iax
        ELg4 = 0.1 * Ig4 + 0.02 * Iax
        ELg5 = 0.1 * Ig5 + 0.02 * Iax
        return {
            "fc": float(fc),
            "ELg3": float(ELg3),
            "ELg4": float(ELg4),
            "ELg5": float(ELg5),
        }

    def _parse_output(self, stdout):
        # Best-effort parsing: try two strategies
        # 1) key: value lines (e.g. 'fc: 1.23')
        # 2) whitespace-separated numeric row containing several outputs
        data = {"fc": None, "ELg3": None, "ELg4": None, "ELg5": None}

        lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
        # strategy A: key:value
        for line in lines:
            for k in data.keys():
                if line.lower().startswith(k.lower() + ":"):
                    try:
                        data[k] = float(line.split(":", 1)[1].strip())
                    except Exception:
                        pass

        # if we already found all values, return
        if all(v is not None for v in data.values()):
            return data

        # strategy B: look for a line with many whitespace-separated numbers
        for line in reversed(lines):
            parts = line.split()
            # attempt to parse floats
            vals = []
            for p in parts:
                try:
                    vals.append(float(p))
                except Exception:
                    vals = []
                    break
            if len(vals) >= 6:
                # mapping (based on observed simulator):
                # column 1 -> fc
                # column 4 -> ELg3
                # column 5 -> ELg4
                # column 6 -> ELg5
                try:
                    data["fc"] = float(vals[0])
                    data["ELg3"] = float(vals[3])
                    data["ELg4"] = float(vals[4])
                    data["ELg5"] = float(vals[5])
                    return data
                except Exception:
                    pass

        # nothing parsed, return data with possible Nones
        return data
