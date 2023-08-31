from src import utils
from src.linear_solver import print_ILP_size, LinearAGV
#from src.linear_solver import solve
import time
import unittest


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
        stations = utils.create_stations_list(tracks)
        J = utils.create_agv_list(agv_routes)

        d_max = {i: 5 for i in J}
        tau_pass = {(agv, way[0], way[1]): 5 for agv, way in agv_routes.items()}
        tau_headway = {(0, 1, "s0", "s1"): 2, (1, 0, "s0", "s1"): 2}
        tau_operation = {(agv, station): 1 for agv in J for station in stations}

        AGV = LinearAGV(M, tracks, tracks_len, agv_routes, d_max, tau_pass, tau_headway, tau_operation, weights,
                        initial_conditions)
        print_ILP_size(AGV.A_ub, AGV.b_ub, AGV.A_eq, AGV.b_eq)

        
        model = AGV.create_linear_model()
        model.print_information()
       

        if False:
            begin = time.time()
            sol = model.solve()
            end = time.time()
            print("time: ", end-begin)
            model.print_solution(print_zeros=True)

            sol = utils.see_variables(res.x, iterators["x"])
            assert sol[('in', 0, 's0')] == 0.
            assert sol[('out', 0, 's0')] == 1.
            assert sol[('in', 0, 's1')] == 6.
            assert sol[('out', 0, 's1')] == 7.

            assert sol[('out', 1, 's0')] == 3.
            assert sol[('in', 1, 's1')] == 8.
            assert sol[('out', 1, 's1')] == 9.

            assert sol[(0, 1, 's1')] == 1. 
            assert sol[(1, 0, 's1')] == 0. 

            assert sol[(0, 1, 's0')] == 1. 
            assert sol[(1, 0, 's0')] == 0. 



    if __name__ == "__main__":
        test_line_fragemnt()