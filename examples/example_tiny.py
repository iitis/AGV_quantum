# 2 AGV example
from dimod.sampleset import SampleSet
import dimod
import minorminer
from dwave.system import DWaveSampler
from src import utils
from src.linear_solver import print_ILP_size, LinearAGV
from src.quadratic_solver import QuadraticAGV
from src.qubo_solver import annealing
import numpy as np
import pickle
import csv
import time
import os
import json


from math import sqrt
from src.LinearProg import LinearProg
from src.process_results import print_results
from src.quadratic_solver_CPLEX import quadratic_solve_qubo, check_solution, save_results
from src.utils import check_solution_list
from src import train_diagram

cwd = os.getcwd()

M = 5
tracks = [("s0", "s1"), ("s1", "s0"),
          ("s1", "s2"), ("s2", "s1")
          ]

# to include track_len into track would require some refactoring
tracks_len = {("s0", "s1"): 6, ("s1", "s0"): 6,
              ("s1", "s2"): 6, ("s2", "s1"): 6,
              }

agv_routes = {0: ("s0", "s1"),
              1: ("s1", "s2")}

stations = utils.create_stations_list(tracks)
J = utils.create_agv_list(agv_routes)
agv_routes_as_edges = utils.agv_routes_as_edges(agv_routes)
all_same_way = utils.create_same_way_dict(agv_routes)

graph = utils.create_graph(tracks, agv_routes)

d_max = {i: 1 for i in J}
tau_pass = {(j, s, sp): tracks_len[(s, sp)] for j in J for s, sp in agv_routes_as_edges[j]}
tau_headway = {(j, jp, s, sp): 2 for (j, jp) in all_same_way.keys() for (s, sp) in all_same_way[(j, jp)]}

tau_operation = {(agv, station): 2 for agv in J for station in stations}

initial_conditions = {("in", 0, "s0"): 0, ("in", 1, "s1"): 7}

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
    "--train_diagram",
    type=int,
    help="Make train diagram for CPLEX solution",
    default=0,
)
parser.add_argument(
    "--solve_quadratic",
    type=int,
    help="Solve using hybrid quantum-classical approach",
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
        if args.train_diagram:
            train_diagram.plot_train_diagram(sol, agv_routes, tracks_len, 3)


    if solve_quadratic:
        model = QuadraticAGV(AGV)
        p = 5
        model.to_bqm_qubo_ising(p)

