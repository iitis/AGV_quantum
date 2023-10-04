# 15 AGVs 7 zones d_max = 40
import pickle
import time
import os

from AGV_quantum import create_stations_list, create_agv_list, create_graph, create_same_way_dict, agv_routes_as_edges
from AGV_quantum import print_ILP_size, LinearAGV
from AGV_quantum import QuadraticAGV
from AGV_quantum import annealing, constrained_solver, hybrid_anneal



cwd = os.getcwd()

M = 50
tracks = [("s0", "s1"), ("s1", "s0"),
          ("s1", "s2"), ("s2", "s1"),
          ("s2", "s3"), ("s3", "s2"),
          ("s3", "s4"), ("s4", "s3"),
          ("s4", "s5"), ("s5", "s4"),
          ("s5", "s6")]

# to include track_len into track would require some refactoring
tracks_len = {("s0", "s1"): 6, ("s1", "s0"): 6,
              ("s1", "s2"): 6, ("s2", "s1"): 6,
              ("s2", "s3"): 0, ("s3", "s2"): 0,
              ("s3", "s4"): 5, ("s4", "s3"): 5,
              ("s4", "s5"): 4, ("s5", "s4"): 4,
              ("s5", "s6"): 4, ("s6", "s5"): 4}

agv_routes = {0: ("s0", "s1", "s2", "s3"), 
              1: ("s0", "s1", "s2", "s3"),
              2: ("s0", "s1", "s2", "s3"), 
              3: ("s0", "s1", "s2", "s3"),
              4: ("s0", "s1", "s2"),
              5: ("s0", "s1", "s2"),
              6: ("s4", "s3", "s2", "s1"),
              7: ("s4", "s3", "s2", "s1", "s0"),
              8: ("s4", "s3", "s2", "s1", "s0"),
              9: ("s4", "s3", "s2", "s1", "s0"),
              10: ("s4", "s3", "s2", "s1", "s0"),
              11: ("s4", "s3", "s2", "s1", "s0"),
              12: ("s2", "s3"),
              13: ("s6", "s5", "s4", "s3"),
              14: ("s5", "s6")
              }

stations = create_stations_list(tracks)
J = create_agv_list(agv_routes)
agv_routes_as_e = agv_routes_as_edges(agv_routes)
all_same_way = create_same_way_dict(agv_routes)
graph = create_graph(tracks, agv_routes)

d_max = {i: 40 for i in J}
tau_pass = {(j, s, sp): tracks_len[(s, sp)] for j in J for s, sp in agv_routes_as_e[j]}
tau_headway = {(j, jp, s, sp): 2 if (s, sp) != ("s2", "s3") and (s, sp) != ("s3", "s2") else 0
               for (j, jp) in all_same_way.keys() for (s, sp) in all_same_way[(j, jp)]}

tau_operation = {(agv, station): 2 for agv in J for station in stations}

initial_conditions = {("in", 0, "s0"): 0, ("in", 1, "s0"): 1, ("in", 2, "s0"): 2, ("in", 3, "s0"): 3,
                      ("in", 4, "s0"): 4, ("in", 5, "s0"): 5, ("in", 6, "s4"): 0, ("in", 7, "s4"): 1,
                      ("in", 8, "s4"): 2, ("in", 9, "s4"): 3, ("in", 10, "s4"): 8, 
                      ("in", 11, "s4"): 5, 
                      ("in", 12, "s2"): 7, ("in", 13, "s6"): 9, ("in", 14, "s5"): 9
                    }

weights = {j: 1 for j in J}
print("prepare ILP")

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

    if solve_quadratic:
        hybrid = "bqm" # select hybrid solver bqm or cqm
        p = 5  # penalty for QUBO creation
        
        model = QuadraticAGV(AGV)
        model.to_bqm_qubo_ising(p)
        model.to_cqm()
        cwd = os.getcwd()
        save_path = os.path.join(cwd, "..", "annealing_results", "15_AGV")
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

