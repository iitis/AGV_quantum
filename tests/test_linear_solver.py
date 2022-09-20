import unittest
import numpy as np
from src import linear_solver
from src import utils


class SingleStation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.M = 50
        cls.tracks = [("s0",)]
        cls.agv_routes = {0: ("s0",), 1: ("s0",)}
        cls.initial_conditions = {("in", 0, "s0"): 0, ("in", 1, "s0"): 0}
        cls.graph = utils.create_graph(cls.tracks, cls.agv_routes)
        cls.iterators = utils.create_iterators(cls.graph, cls.agv_routes)

    def test_preference_variables_y_single(self):
        PVY, PVY_b = linear_solver.create_precedence_matrix_y(self.iterators)
        np.testing.assert_array_equal(PVY, np.array([[0, 0, 0, 0, 1, 1]]))
        self.assertEqual(PVY_b, np.array([1]))

    def test_create_bounds_single(self):
        bounds = linear_solver.create_bounds(self.initial_conditions,  self.iterators)
        self.assertEqual(bounds, [(0, None), (0, None), (0, None), (0, None), (0, 1), (0, 1)])


class MultipleStationsNoOpposite(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.M = 50
        cls.tracks = [("s0", "s1"), ("s1", "s0"), ("s0", "s2"), ("s2", "s3")]
        cls.agv_routes = {0: ("s0", "s1"), 1: ("s0", "s2", "s3")}
        cls.graph = utils.create_graph(cls.tracks, cls.agv_routes)
        cls.iterators = utils.create_iterators(cls.graph, cls.agv_routes)
        cls.initial_conditions = {("in", 0, "s0"): 0, ("in", 1, "s0"): 1}

    def test_preference_variables_y_multi(self):
        PVY, PVY_b = linear_solver.create_precedence_matrix_y(self.iterators)
        # TO DO later
        #np.testing.assert_array_equal(PVY, np.array([[0, 0, 0, 0, 1, 1]]))
        #self.assertEqual(PVY_b, np.array([1]))

    def test_create_bounds_multi(self):
        bounds = linear_solver.create_bounds(self.initial_conditions,  self.iterators)
        self.assertEqual(bounds, [(0, None), (0, None), (1, None), (0, None), (0, None), (0, None), (0, None),
                                  (0, None), (0, None), (0, None), (0, 1), (0, 1)])

#if __name__ == '__main__':
#    unittest.main()
