"""Unit tests for NSGA-II core utilities: sorting, crowding, selection, and constraints."""

import pytest
import numpy as np
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nsga2 import (
    fast_nondominated_sort,
    crowding_distance,
    tournament_selection,
    repair_gears,
)


class TestFastNondominatedSort:
    """Test non-dominated sorting."""

    def test_single_front(self):
        """Test non-dominated solutions in front 0."""
        objs = [(1.0, 4.0), (2.0, 3.0), (3.0, 2.0), (4.0, 1.0)]
        fronts = fast_nondominated_sort(objs)
        
        assert len(fronts) == 1
        assert set(fronts[0]) == {0, 1, 2, 3}

    def test_multiple_fronts(self):
        """Test dominated solutions in separate fronts."""
        objs = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
        fronts = fast_nondominated_sort(objs)
        
        assert len(fronts) == 3
        assert fronts[0] == [0]
        assert fronts[1] == [1]
        assert fronts[2] == [2]


class TestCrowdingDistance:
    """Test crowding distance for diversity."""

    def test_boundary_infinite(self):
        """Test boundary solutions have infinite distance."""
        objs = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
        front = [0, 1, 2]
        distances = crowding_distance(objs, front)
        
        assert distances[0] == float("inf")
        assert distances[2] == float("inf")
        assert 0 < distances[1] < float("inf")

    def test_two_solutions(self):
        """Test two solutions both get infinite distance."""
        objs = [(1.0, 2.0), (3.0, 4.0)]
        front = [0, 1]
        distances = crowding_distance(objs, front)
        
        assert distances[0] == float("inf")
        assert distances[1] == float("inf")


class TestTournamentSelection:
    """Test tournament selection."""

    def test_dominance_selection(self):
        """Test dominated solution loses."""
        random.seed(42)
        pop = [np.array([4.0, 0.45, 2.0, 1.5, 1.0]), np.array([3.0, 0.40, 1.0, 1.0, 0.8])]
        objs = [(100.0, -0.5), (200.0, -0.3)]
        
        selections = [tournament_selection(pop, objs) for _ in range(50)]
        first_count = sum(1 for s in selections if np.array_equal(s, pop[0]))
        
        assert first_count > 0


class TestRepairGears:
    """Test gear constraint ig3 >= ig4 >= ig5."""

    def test_correct_order_unchanged(self):
        """Test valid order remains unchanged."""
        x = np.array([4.0, 0.45, 2.0, 1.5, 1.0])
        result = repair_gears(x, [2, 3, 4], (0.5, 2.25))
        
        np.testing.assert_array_almost_equal(result, x)
        assert result[2] >= result[3] >= result[4]

    def test_reverse_order_fixed(self):
        """Test reverse order is corrected."""
        x = np.array([4.0, 0.45, 1.0, 1.5, 2.0])
        result = repair_gears(x, [2, 3, 4], (0.5, 2.25))
        
        assert result[2] >= result[3] >= result[4]
        np.testing.assert_array_almost_equal(result[2:5], [2.0, 1.5, 1.0])

    def test_bounds_clipping(self):
        """Test out-of-bounds values are clipped."""
        x = np.array([4.0, 0.45, 3.0, 2.5, 0.2])
        result = repair_gears(x, [2, 3, 4], (0.5, 2.25))
        
        assert result[2] <= 2.25
        assert result[4] >= 0.5
        assert result[2] >= result[3] >= result[4]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
