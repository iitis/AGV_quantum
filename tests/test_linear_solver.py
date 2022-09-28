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
        cls.initial_conditions = {("in", 0, "s0"): 1, ("in", 1, "s0"): 2}
        cls.graph = utils.create_graph(cls.tracks, cls.agv_routes)
        cls.iterators = utils.create_iterators(cls.graph, cls.agv_routes)
        cls.d_max = {j: 10 for j in cls.agv_routes.keys()}

    def test_preference_variables_y_single(self):
        PVY, PVY_b = linear_solver.create_precedence_matrix_y(self.iterators)
        np.testing.assert_array_equal(PVY, np.array([[0, 0, 0, 0, 1, 1]]))
        self.assertEqual(PVY_b, np.array([1]))

    # def test_create_bounds_single(self): TODO

    def test_single_line_matrix_single(self):
        SL, SL_b = linear_solver.create_single_line_matrix(self.M, self.iterators)
        self.assertTrue(np.array_equal(SL, np.array([])))


class MultipleStationsNoOpposite(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.M = 50
        cls.tracks = [("s0", "s1"), ("s1", "s0"), ("s0", "s2"), ("s2", "s3")]
        cls.agv_routes = {0: ("s0", "s1"), 1: ("s0", "s2", "s3"), 2: ("s2", "s3")}
        cls.graph = utils.create_graph(cls.tracks, cls.agv_routes)
        cls.iterators = utils.create_iterators(cls.graph, cls.agv_routes)
        cls.initial_conditions = {("in", 0, "s0"): 1, ("in", 1, "s0"): 3, ("in", 2, "s2"): 0}
        cls.d_max = {j: 100 for j in cls.agv_routes.keys()}
        cls.weights = {j: 1 for j in cls.agv_routes.keys()}
        cls.tracks_len = {("s0", "s1"): 1, ("s1", "s0"): 1, ("s0", "s2"): 2, ("s2", "s3"): 2}

        all_same_way = utils.create_same_way_dict(cls.agv_routes)
        J = utils.create_agv_list(cls.agv_routes)
        stations = utils.create_stations_list(cls.tracks)
        agv_routes_as_edges = utils.agv_routes_as_edges(cls.agv_routes)

        cls.tau_headway = {(j, jp, s, sp): 2 for (j, jp) in all_same_way.keys() for (s, sp) in all_same_way[(j, jp)]}
        cls.tau_pass = {(j, s, sp): cls.tracks_len[(s, sp)] for j in J for s, sp in agv_routes_as_edges[j]}
        cls.tau_operation = {(agv, station): 2 for agv in J for station in stations}

        cls.x_iter = cls.iterators["x"]

    def test_tau_headway(self):
        self.assertEqual(self.tau_headway, {(1, 2, "s2", "s3"): 2, (2, 1, "s2", "s3"): 2})

    # def test_create_bounds_multi(self): TODO

    def test_preference_variables_y_multi(self):
        PVY, PVY_b = linear_solver.create_precedence_matrix_y(self.iterators)
        self.assertEqual(PVY.shape, (3, len(self.x_iter)))
        equations_list = [utils.see_non_zero_variables(PVY[i], self.x_iter) for i in range(PVY.shape[0])]
        self.assertIn({(1, 0, "s0"): 1, (0, 1, "s0"): 1}, equations_list)
        self.assertIn({(1, 2, "s2"): 1, (2, 1, "s2"): 1}, equations_list)
        self.assertIn({(1, 2, "s3"): 1, (2, 1, "s3"): 1}, equations_list)
        self.assertTrue(np.array_equal(PVY_b, np.array([1 for _ in range(PVY.shape[0])])))

    def test_headway_matrix(self):
        MH, MH_b = linear_solver.create_minimal_headway_matrix(self.M, self.tracks, self.agv_routes,
                                                               self.tau_headway, self.iterators)
        self.assertEqual(MH.shape, (2, len(self.x_iter)))
        equations_list = [utils.see_non_zero_variables(MH[i], self.x_iter) for i in range(MH.shape[0])]
        self.assertIn({("out", 1, "s2"): 1, ("out", 2, "s2"): -1, (2, 1, "s2"): -1 * self.M}, equations_list)
        self.assertIn({("out", 1, "s2"): -1, ("out", 2, "s2"): 1, (1, 2, "s2"): -1 * self.M}, equations_list)
        self.assertTrue(np.array_equal(MH_b, np.array([-1 * tau_h for tau_h in self.tau_headway.values()])))

    def test_passing_time_matrix(self):
        MPT, MPT_b = linear_solver.create_minimal_passing_time_matrix(self.agv_routes, self.tau_pass, self.iterators)
        self.assertEqual(MPT.shape, (4, len(self.x_iter)))
        equations_list = [utils.see_non_zero_variables(MPT[i], self.x_iter) for i in range(MPT.shape[0])]
        self.assertIn({("out", 0, "s0"): 1, ("in", 0, "s1"): -1}, equations_list)
        self.assertIn({("out", 1, "s0"): 1, ("in", 1, "s2"): -1}, equations_list)
        self.assertIn({("out", 1, "s2"): 1, ("in", 1, "s3"): -1}, equations_list)
        self.assertIn({("out", 2, "s2"): 1, ("in", 2, "s3"): -1}, equations_list)
        self.assertTrue(np.array_equal(MPT_b, np.array([-1 * tau_p for tau_p in self.tau_pass.values()])))

    def test_junction_condition_matrix(self):
        JC, JC_b = linear_solver.create_junction_condition_matrix(self.M, self.tracks, self.agv_routes,
                                                                  self.tau_operation, self.iterators)
        self.assertEqual(len(self.iterators["y"]), 6)
        self.assertEqual(JC.shape, (13, len(self.x_iter)))

    def test_single_line_matrix(self):
        SL, SL_b = linear_solver.create_single_line_matrix(self.M, self.iterators)
        self.assertTrue(np.array_equal(SL, np.array([])))

    def test_no_overtake_matrix(self):
        NO, NO_b = linear_solver.create_no_overtake_matrix(self.agv_routes, self.tau_headway, self.iterators)
        self.assertEqual(NO.shape, (2, len(self.x_iter)))
        equations_list = [utils.see_non_zero_variables(NO[i], self.x_iter) for i in range(NO.shape[0])]
        self.assertIn({(1, 2, 's2'): 1, (1, 2, 's3'): -1}, equations_list)
        self.assertIn({(2, 1, 's2'): 1, (2, 1, 's3'): -1}, equations_list)
        self.assertTrue(np.array_equal(NO_b, np.array([0 for _ in range(NO.shape[0])])))


    def test_solve(self):
        c, A_ub, b_ub, A_eq, b_eq, bounds, iterators = linear_solver.make_linear_problem(self.M, self.tracks, self. tracks_len, self.agv_routes, self.d_max,
                                          self.tau_pass, self.tau_headway, self.tau_operation,
                                          self.weights, self.initial_conditions)
        res, iterators = linear_solver.solve(c, A_ub, b_ub, A_eq, b_eq, bounds, iterators)
        self.assertTrue(res.success)


class TwoStationsOpposite(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.M = 50
        cls.tracks = [("s0", "s1")]
        cls.agv_routes = {0: ("s0", "s1"), 1: ("s1", "s0")}
        cls.tracks_len = {("s0", "s1"): 1, ("s1", "s0"): 1}
        cls.graph = utils.create_graph(cls.tracks, cls.agv_routes)
        cls.iterators = utils.create_iterators(cls.graph, cls.agv_routes)

        cls.initial_conditions = {("in", 0, "s0"): 0, ("in", 1, "s0"): 0}
        cls.d_max = {j: 100 for j in cls.agv_routes.keys()}
        cls.weights = {j: 1 for j in cls.agv_routes.keys()}

        all_same_way = utils.create_same_way_dict(cls.agv_routes)
        J = utils.create_agv_list(cls.agv_routes)
        stations = utils.create_stations_list(cls.tracks)
        agv_routes_as_edges = utils.agv_routes_as_edges(cls.agv_routes)

        cls.tau_headway = {(j, jp, s, sp): 2 for (j, jp) in all_same_way.keys() for (s, sp) in all_same_way[(j, jp)]}
        cls.tau_pass = {(j, s, sp): cls.tracks_len[(s, sp)] for j in J for s, sp in agv_routes_as_edges[j]}
        cls.tau_operation = {(agv, station): 2 for agv in J for station in stations}

        cls.x_iter = cls.iterators["x"]

    def test_equations(self):
        PVY, PVY_b = linear_solver.create_precedence_matrix_y(self.iterators)
        PVZ, PVZ_b = linear_solver.create_precedence_matrix_z(self.iterators)
        NO, NO_b = linear_solver.create_no_overtake_matrix(self.agv_routes, self.tau_headway, self.iterators)

        MPT, MPT_b = linear_solver.create_minimal_passing_time_matrix(self.agv_routes, self.tau_pass, self.iterators)
        MH, MH_b = linear_solver.create_minimal_headway_matrix(self.M, self.tracks, self.agv_routes, self.tau_headway, self.iterators)
        JC, JC_b = linear_solver.create_junction_condition_matrix(self.M, self.tracks, self.agv_routes, self.tau_operation, self.iterators)
        SL, SL_b = linear_solver.create_single_line_matrix(self.M, self.iterators)

        if MPT.size >= 2 and MH.size >= 2:  # TO DO more sensible, for now is hack
            if SL.size > 0:
                A_ub = np.concatenate((MPT, MH, JC, SL))
                b_ub = np.concatenate((MPT_b, MH_b, JC_b, SL_b))
            else:
                A_ub = np.concatenate((MPT, MH, JC))
                b_ub = np.concatenate((MPT_b, MH_b, JC_b))
        else:
            A_ub = JC
            b_ub = JC_b

        if NO.size > 0:
            if PVZ.size > 0:
                A_eq = np.concatenate((PVY, PVZ, NO))
                b_eq = np.concatenate((PVY_b, PVZ_b, NO_b))
            else:
                A_eq = np.concatenate((PVY, NO))
                b_eq = np.concatenate((PVY_b, NO_b))
        else:
            A_eq = PVY
            b_eq = PVY_b

        for i in range(A_eq.shape[0]):
            print(utils.see_non_zero_variables(A_eq[i], self.x_iter))
        print(b_eq)
        print(A_eq)

class OneSameWayOneOpposite(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.M = 50
        cls.tracks = [("s0", "s1"), ("s1", "s2")]
        cls.agv_routes = {0: ("s0", "s1"), 1: ("s0", "s1", "s2"), 2: ("s2", "s1")}
        cls.graph = utils.create_graph(cls.tracks, cls.agv_routes)
        cls.iterators = utils.create_iterators(cls.graph, cls.agv_routes)
        cls.d_max = {j: 10 for j in cls.agv_routes.keys()}
        cls.weights = {j:1 for j in cls.agv_routes.keys()}
        cls.tracks_len = {("s0", "s1"): 1, ("s1", "s2"): 2, ("s2", "s1"): 2}

        all_same_way = utils.create_same_way_dict(cls.agv_routes)
        J = utils.create_agv_list(cls.agv_routes)
        stations = utils.create_stations_list(cls.tracks)
        agv_routes_as_edges = utils.agv_routes_as_edges(cls.agv_routes)

        cls.tau_headway = {(j, jp, s, sp): 2 for (j, jp) in all_same_way.keys() for (s, sp) in all_same_way[(j, jp)]}
        cls.tau_pass = {(j, s, sp): cls.tracks_len[(s, sp)] for j in J for s, sp in agv_routes_as_edges[j]}
        cls.tau_operation = {(agv, station): 2 for agv in J for station in stations}

        cls.x_iter = cls.iterators["x"]

    def test_tau_headway(self):

        self.assertEqual(self.tau_headway, {(0, 1, "s0", "s1"): 2, (1, 0, "s0", "s1"): 2})

    def test_headway_matrix(self):
        MH, MH_b = linear_solver.create_minimal_headway_matrix(self.M, self.tracks, self.agv_routes,
                                                               self.tau_headway, self.iterators)
        self.assertEqual(MH.shape, (2, len(self.x_iter)))
        equations_list = [utils.see_non_zero_variables(MH[i], self.x_iter) for i in range(MH.shape[0])]
        self.assertIn({("out", 0, "s0"): 1, ("out", 1, "s0"): -1, (1, 0, "s0"): -1 * self.M}, equations_list)
        self.assertIn({("out", 0, "s0"): -1, ("out", 1, "s0"): 1, (0, 1, "s0"): -1 * self.M}, equations_list)
        self.assertTrue(np.array_equal(MH_b, np.array([-1 * tau_h for tau_h in self.tau_headway.values()])))

    def test_single_line_matrix(self):
        SL, SL_b = linear_solver.create_single_line_matrix(self.M, self.iterators)
        self.assertEqual(SL.shape, (2, len(self.x_iter)))
        self.assertTrue(np.array_equal(SL_b, np.array([0 for _ in range(SL.shape[0])])))
        equations_list = [utils.see_non_zero_variables(SL[i], self.x_iter) for i in range(SL.shape[0])]
        self.assertIn({("in", 1, "s2"): 1, ("out", 2, "s2"): -1, (2, 1, "s2", "s1"): -1 * self.M}, equations_list)
        self.assertIn({('in', 2, 's1'): 1, ('out', 1, 's1'): -1, (1, 2, 's1', 's2'): -1 * self.M}, equations_list)


    def test_no_overtake_matrix(self):
        NO, NO_b = linear_solver.create_no_overtake_matrix(self.agv_routes, self.tau_headway, self.iterators)


    def test_solve(self):
        c, A_ub, b_ub, A_eq, b_eq, bounds, iterators = linear_solver.make_linear_problem(self.M, self.tracks, self. tracks_len, self.agv_routes, self.d_max,
                                          self.tau_pass, self.tau_headway, self.tau_operation,
                                          self.weights, initial_conditions={})
        res, iterators = linear_solver.solve(c, A_ub, b_ub, A_eq, b_eq, bounds, iterators)

        self.assertTrue(res.success)



class TestZeroDistance(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
