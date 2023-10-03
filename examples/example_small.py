# 4 AGV 5 zoneas d_max = 10 example
import pickle
import time
import os

from src import create_stations_list, create_agv_list, create_graph, create_same_way_dict, agv_routes_as_edges
from src import plot_train_diagram
from src import print_ILP_size, LinearAGV
from src import QuadraticAGV
from src import annealing, constrained_solver, hybrid_anneal


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

stations = create_stations_list(tracks)
J = create_agv_list(agv_routes)
agv_routes_as_e = agv_routes_as_edges(agv_routes)
all_same_way = create_same_way_dict(agv_routes)

graph = create_graph(tracks, agv_routes)

d_max = {i: 10 for i in J}
tau_pass = {(j, s, sp): tracks_len[(s, sp)] for j in J for s, sp in agv_routes_as_e[j]}
tau_headway = {(j, jp, s, sp): 2 if (s, sp) != ("s2", "s3") and (s, sp) != ("s3", "s2") else 0
               for (j, jp) in all_same_way.keys() for (s, sp) in all_same_way[(j, jp)]}

tau_operation = {(agv, station): 2 for agv in J for station in stations}

initial_conditions = {("in", 0, "s0"): 0, ("in", 1, "s1"): 8, ("in", 2, "s4"): 8, ("in", 4, "s2"): 15}

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
            plot_train_diagram(sol, agv_routes, tracks_len, 5)

    if solve_quadratic:
        hybrid = "bqm" # select hybrid solver bqm or cqm
        p = 5 # penalty for QUBO creation

        model = QuadraticAGV(AGV)
        model.to_bqm_qubo_ising(p)
        model.to_cqm()
        cwd = os.getcwd()
        save_path = os.path.join(cwd, "..", "annealing_results", "4_AGV")
        cqm = model.cqm
        bqm = model.bqm
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
