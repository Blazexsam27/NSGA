"""Integration tests for NSGA-II with simulator."""

import pytest
import numpy as np
import os
import sys
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nsga2 import run_nsga, dominates
from simulator import Simulator


class TestNSGAIntegration:
    """Integration tests for complete NSGA-II pipeline."""

    def test_nsga_run_with_mock_simulator(self, tmp_path):
        """Test full NSGA run with mock simulator."""
        original_dir = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            pareto = run_nsga(
                pop_size=10,
                generations=5,
                seed=42,
                executable_path=None
            )
            
            assert len(pareto) > 0, "Pareto front should not be empty"
            assert len(pareto) <= 10, "Pareto front size should not exceed population"
            
            for ind, obj in pareto:
                assert ind.shape == (5,), "Individual should have 5 parameters"
                assert len(obj) == 2, "Should have 2 objectives"
                assert isinstance(obj[0], float), "fc should be float"
                assert isinstance(obj[1], float), "neg_avgEL should be float"
        finally:
            os.chdir(original_dir)

    def test_pareto_front_non_dominated(self, tmp_path):
        """Test that Pareto front contains only non-dominated solutions."""
        original_dir = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            pareto = run_nsga(pop_size=15, generations=5, seed=123)
            objs = [obj for _, obj in pareto]
            
            for i, obj_i in enumerate(objs):
                for j, obj_j in enumerate(objs):
                    if i != j:
                        assert not dominates(obj_i, obj_j), \
                            f"Solution {i} dominates solution {j} in Pareto front"
        finally:
            os.chdir(original_dir)

    def test_pareto_csv_output(self):
        """Test that Pareto front CSV is created with correct format."""
        # Run NSGA (saves to module directory)
        run_nsga(pop_size=10, generations=3, seed=42)
        
        # CSV is saved in the nsga2.py directory
        nsga_dir = os.path.dirname(os.path.abspath(sys.modules['nsga2'].__file__))
        csv_path = os.path.join(nsga_dir, "pareto_front.csv")
        
        assert os.path.exists(csv_path), "pareto_front.csv should be created"
        
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == ["Iax", "Rtr", "Ig3", "Ig4", "Ig5", "fc", "neg_avgEL"]
            
            rows = list(reader)
            assert len(rows) > 0, "CSV should contain solutions"
            
            for row in rows:
                assert len(row) == 7, "Each row should have 7 values"
                for val in row:
                    float(val)

    def test_constraints_satisfied(self, tmp_path):
        """Test that all solutions satisfy constraints."""
        original_dir = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            pareto = run_nsga(pop_size=20, generations=5, seed=42)
            bounds = [(3.0, 5.5), (0.4, 0.5), (0.5, 2.25), (0.5, 2.25), (0.5, 2.25)]
            
            for ind, _ in pareto:
                for i, (lo, hi) in enumerate(bounds):
                    assert lo <= ind[i] <= hi, \
                        f"Parameter {i} = {ind[i]} violates bounds [{lo}, {hi}]"
                
                assert ind[2] >= ind[3], f"ig3={ind[2]} should be >= ig4={ind[3]}"
                assert ind[3] >= ind[4], f"ig4={ind[3]} should be >= ig5={ind[4]}"
        finally:
            os.chdir(original_dir)

    def test_simulator_mock_mode(self):
        """Test simulator in mock mode."""
        sim = Simulator(executable_path="nonexistent.exe", strict=False)
        
        assert sim.use_mock is True, "Should use mock when executable not found"
        
        x = np.array([4.0, 0.45, 2.0, 1.5, 1.0])
        result = sim.evaluate(x)
        
        assert "fc" in result
        assert "ELg3" in result
        assert "ELg4" in result
        assert "ELg5" in result
        assert all(isinstance(v, float) for v in result.values())


class TestSimulatorIntegration:
    """Integration tests for Simulator class."""

    def test_simulator_strict_mode_raises(self):
        """Test that strict mode raises error when executable not found."""
        with pytest.raises(FileNotFoundError):
            Simulator(executable_path="nonexistent.exe", strict=True)

    def test_invalid_parameters_handled(self):
        """Test that invalid input parameters are handled properly."""
        sim = Simulator()
        
        # Test wrong shape
        with pytest.raises(ValueError):
            sim.evaluate(np.array([1.0, 2.0]))  # Only 2 values instead of 5
        
        with pytest.raises(ValueError):
            sim.evaluate(np.array([[1.0, 2.0, 3.0, 4.0, 5.0]]))  # 2D instead of 1D


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
