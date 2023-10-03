# 2 AGV and 4 zones d_max = 10
import dimod

from src import utils
from src.linear_solver import print_ILP_size, LinearAGV
from src.quadratic_solver import QuadraticAGV
from src.qubo_solver import annealing, constrained_solver, hybrid_anneal
from pathlib import Path
import numpy as np
import pickle
import csv
import time
import os

from src.LinearProg import LinearProg
from src.process_results import print_results
from math import sqrt, log10

M = 20
tracks = [("s0", "s1"), ("s1", "s0"),
          ("s1", "s2"), ("s2", "s1"),
          ("s2", "s3"), ("s3", "s2")
          ]

# to include track_len into track would require some refactoring
tracks_len = {("s0", "s1"): 6, ("s1", "s0"): 6,
              ("s1", "s2"): 6, ("s2", "s1"): 6,
              ("s2", "s3"): 0, ("s3", "s2"): 0
            }

agv_routes = {0: ("s1", "s2", "s3"),
              1: ("s0", "s1", "s2")
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

initial_conditions = {("in", 0, "s1"): 8, ("in", 1, "s0"): 0
                     }

weights = {j: 1 for j in J}

import argparse
parser = argparse.ArgumentParser("Solve linear or quadratic") 
parser.add_argument(
    "--solve_linear",
    type=int,
    help="Solve the problem on CPLEX",
    default=1,
)
parser.add_argument(
    "--solve_quadratic",
    type=int,
    help="Solve via QUBO approach",
    default=0,
)

args = parser.parse_args()

solve_linear = args.solve_linear
solve_quadratic = args.solve_quadratic

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
        model.to_cqm()
        cwd = os.getcwd()
        save_path = os.path.join(cwd, "..", "annealing_results", "2_AGV")
        cqm = model.cqm
        bqm = model.bqm
        hybrid = "bqm"
        if hybrid == "cqm":
            sampleset = constrained_solver(cqm)
        elif hybrid == "bqm":
            sampleset = hybrid_anneal(bqm)
        else:
            sampleset = 0  # To implement
        info = sampleset.info
        print(sampleset)
        print(info)

        with open(os.path.join(save_path, f"new_{hybrid}_info.pkl"), "wb") as f:
            pickle.dump(info, f)

        with open(os.path.join(save_path, f"new_{hybrid}.pkl"), "wb") as f:
            pickle.dump(sampleset.to_serializable(), f)