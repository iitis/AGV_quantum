from typing import Callable, Dict, List, Tuple
import time
import unittest

from AGV_quantum import create_stations_list, create_agv_list
from AGV_quantum import print_ILP_size, LinearAGV, QuadraticAGV



#cwd = os.getcwd()

class TestExample(unittest.TestCase):

    @classmethod

    def test_line_fragemnt(self):
        """
            s0                           s1
        1 -> 
        0 ->  --------------------------
        """

        M = 50
        tracks = [("s0", "s1"), ("s1", "s0")]
        agv_routes = {0: ("s0", "s1"), 1: ("s0", "s1")}
        weights = {0: 1, 1: 1}
        tracks_len = {("s0", "s1"): 1, ("s1", "s0"): 1}

        initial_conditions = {("in", 0, "s0"): 0, ("in", 1, "s0"): 1}
        stations = create_stations_list(tracks)
        J = create_agv_list(agv_routes)

        d_max = {i: 5 for i in J}
        tau_pass = {(agv, way[0], way[1]): 5 for agv, way in agv_routes.items()}
        tau_headway = {(0, 1, "s0", "s1"): 2, (1, 0, "s0", "s1"): 2}
        tau_operation = {(agv, station): 1 for agv in J for station in stations}

        AGV = LinearAGV(M, tracks, tracks_len, agv_routes, d_max, tau_pass, tau_headway, tau_operation, weights,
                        initial_conditions)
        print_ILP_size(AGV.A_ub, AGV.b_ub, AGV.A_eq, AGV.b_eq)

        
        model = AGV.create_linear_model()
        model.print_information()
       
        
        try:
            begin = time.time()
            sol = model.solve()
            end = time.time()
            print("time: ", end-begin)
            model.print_solution(print_zeros=True)
            is_cplex = True
        except:
            is_cplex = False

        if is_cplex:
            assert sol["y_0_1_s0"] == 1. 
            assert sol["y_0_1_s1"] == 1. 

            assert sol["t_in_0_s0"] == 0.
            assert sol["t_out_0_s0"] == 1.
            assert sol["t_in_0_s1"] == 6.
            assert sol["t_out_0_s1"] == 7.

        # quadratic testing
        model_q = QuadraticAGV(AGV)
        p = 5.
        model_q.to_bqm_qubo_ising(p)
        assert model_q.qubo[0][('x_0[0]', 'x_0[0]')] == -10.0
        model_q.to_cqm()
        print(model_q.cqm.constraints['eq_0'])


    if __name__ == "__main__":
        test_line_fragemnt()