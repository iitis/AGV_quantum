# 4 AGV example

from src import utils
from src.linear_solver import print_ILP_size, LinearAGV
from src.quadratic_solver import QuadraticAGV
from src.qubo_solver import annealing
import numpy as np
import pickle
import time
import os

from scipy.optimize import linprog
from src.LinearProg import LinearProg
from src.process_results import get_results, load_results, print_results, store_result
from math import sqrt

cwd = os.getcwd()

M = 28
tracks = [("s0", "s1"), ("s1", "s0"),
          ("s1", "s2"), ("s2", "s1"),
          ("s2", "s3"), ("s3", "s2"),
          ("s3", "s4"), ("s4", "s3")]

# to include track_len into track would require some refactoring
tracks_len = {("s0", "s1"): 6, ("s1", "s0"): 6,
              ("s1", "s2"): 6, ("s2", "s1"): 6,
              ("s2", "s3"): 0, ("s3", "s2"): 0,
              ("s3", "s4"): 5, ("s4", "s3"): 5
              }

agv_routes = {0: ("s0", "s1", "s2"),
              1: ("s1", "s2"),
              2: ("s4", "s3", "s2"),
              4: ("s2", "s3")
              }

stations = utils.create_stations_list(tracks)
J = utils.create_agv_list(agv_routes)
agv_routes_as_edges = utils.agv_routes_as_edges(agv_routes)
all_same_way = utils.create_same_way_dict(agv_routes)

graph = utils.create_graph(tracks, agv_routes)

d_max = {i: 10 for i in J}
tau_pass = {(j, s, sp): tracks_len[(s, sp)] for j in J for s, sp in agv_routes_as_edges[j]}
tau_headway = {(j, jp, s, sp): 2 if (s, sp) != ("s2", "s3") and (s, sp) != ("s3", "s2") else 0
               for (j, jp) in all_same_way.keys() for (s, sp) in all_same_way[(j, jp)]}

tau_operation = {(agv, station): 2 for agv in J for station in stations}

initial_conditions = {("in", 0, "s0"): 0, ("in", 1, "s1"): 8, ("in", 2, "s4"): 8, ("in", 4, "s2"): 15}

weights = {j: 1 for j in J}

solve_linear = False
solve_quadratic = True

if __name__ == "__main__":

    AGV = LinearAGV(M, tracks, tracks_len, agv_routes, d_max, tau_pass, tau_headway, tau_operation, weights,
                    initial_conditions)
    print_ILP_size(AGV.A_ub, AGV.b_ub, AGV.A_eq, AGV.b_eq)

    if solve_linear:
        model = AGV.create_linear_model()
        model.print_information()
        begin = time.time()
        sol = model.solve()
        end = time.time()
        print("time: ", end-begin)
        model.print_solution(print_zeros=True)
        # AGV.nice_print(model, sol) <- WIP

    if solve_quadratic:
        model = QuadraticAGV(AGV)
        p = 5
        model.to_bqm_qubo_ising(p)

