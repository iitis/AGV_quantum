# 2 AGV example

from src import utils
from src.linear_solver import solve
from src.linear_solver import make_linear_problem
from src.linear_solver import print_ILP_size
from src.qubo_solver import annealing
import numpy as np
import pickle
import csv

from src.LinearProg import LinearProg
from src.process_results import print_results


M = 10
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

agv_routes = {0: ("s0", "s1"),
              1: ("s0", "s1", "s2")
            }

stations = utils.create_stations_list(tracks)
J = utils.create_agv_list(agv_routes)
agv_routes_as_edges = utils.agv_routes_as_edges(agv_routes)
all_same_way = utils.create_same_way_dict(agv_routes)

graph = utils.create_graph(tracks, agv_routes)

d_max = {i: 5 for i in J}
tau_pass = {(j, s, sp): tracks_len[(s, sp)] for j in J for s, sp in agv_routes_as_edges[j]}
tau_headway = {(j, jp, s, sp): 2 if (s, sp) != ("s2", "s3") and (s, sp) != ("s3", "s2") else 0
               for (j, jp) in all_same_way.keys() for (s, sp) in all_same_way[(j, jp)]}

tau_operation = {(agv, station): 2 for agv in J for station in stations}

initial_conditions = {("in", 0, "s0"): 0, ("in", 1, "s0"): 0
                     }

weights = {j: 1 for j in J}

obj, A_ub, b_ub, A_eq, b_eq, bounds, iterators = make_linear_problem(M, tracks, tracks_len, agv_routes, d_max,
                                                 tau_pass, tau_headway, tau_operation, weights, initial_conditions)

res, iterators = solve(obj, A_ub, b_ub, A_eq, b_eq, bounds, iterators)
print_ILP_size(A_ub, b_ub, A_eq, b_eq)

# linear solver
if res.success:
    v_in, v_out = utils.create_v_in_out(tracks_len, agv_routes, tau_operation, iterators, initial_conditions)
    utils.nice_print(res, agv_routes, weights, d_max,  v_in, v_out, iterators)  
else:
    print(res.message)


# QUBO
lp = LinearProg(c=obj, bounds=bounds, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)
p = 2.75

with open("lp_tiny.pkl", "wb") as f:
    pickle.dump(lp, f)

lp._to_bqm_qubo_ising(p)
lp._to_cqm()


# this is QUBO
with open("qubo_tiny.pkl", "wb") as f:
    pickle.dump(lp.qubo, f)



print("-----------------------------------------------------")
print("Number of q-bits", lp._count_qubits())
print("Number of couplings Js:", lp._count_quadratic_couplings())
print("Number of local filds hs:", lp._count_linear_fields())


sdict={"num_sweeps":1_000, "num_reads":500, "beta_range":(0.01, 20)}
dict_list = annealing(lp, "sim", "2_tiny_AGV", sim_anneal_var_dict=sdict, load=False, store=False)
print("Simulated annealing results")
print_results(dict_list)


"""
dict_list = annealing(lp, "cqm", "2_tiny_AGV", load=False, store=False)
print("CQM results:")
print_results(dict_list)


dict_list = annealing(lp, "hyb", "2_tiny_AGV", load=False, store=True)
print("QPU results")
print_results(dict_list)

dict_list = annealing(lp, "real", "2_tiny_AGV", load=True, store=False)
print("QPU results")
print_results(dict_list)
"""