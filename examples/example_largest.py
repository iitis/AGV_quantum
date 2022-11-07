from src import utils
from src.linear_solver import solve
from src.linear_solver import make_linear_problem, create_linear_model
from src.linear_solver import print_ILP_size
from src.qubo_solver import annealing
import pickle
import time

from src.LinearProg import LinearProg
from src.process_results import print_results


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

stations = utils.create_stations_list(tracks)
J = utils.create_agv_list(agv_routes)
agv_routes_as_edges = utils.agv_routes_as_edges(agv_routes)
all_same_way = utils.create_same_way_dict(agv_routes)

graph = utils.create_graph(tracks, agv_routes)

d_max = {i: 40 for i in J}
tau_pass = {(j, s, sp): tracks_len[(s, sp)] for j in J for s, sp in agv_routes_as_edges[j]}
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
obj, A_ub, b_ub, A_eq, b_eq, bounds, iterators = make_linear_problem(M, tracks, tracks_len, agv_routes, d_max,
                                                 tau_pass, tau_headway, tau_operation, weights, initial_conditions)


print_ILP_size(A_ub, b_ub, A_eq, b_eq)
solve_linear = False

if solve_linear:
    print("start solving")
    res, iterators = solve(obj, A_ub, b_ub, A_eq, b_eq, bounds, iterators)

    # linear solver
    if res.success:
        v_in, v_out = utils.create_v_in_out(tracks_len, agv_routes, tau_operation, iterators, initial_conditions)
        utils.nice_print(res, agv_routes, weights, d_max,  v_in, v_out, iterators)
    else:
        print(res.message)


model = create_linear_model(obj, A_ub, b_ub, A_eq, b_eq, bounds, iterators)
model.print_information()
begin = time.time()
s = model.solve()
end = time.time()
print("time: ", end-begin)
model.print_solution(print_zeros=True)
print(model.solve_details)

#
#
# # QUBO
# print("make qubo")
# lp = LinearProg(c=obj, bounds=bounds, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)
# p =2.75
#
# with open("lp_large.pkl", "wb") as f:
#     pickle.dump(lp, f)
#
# lp._to_bqm_qubo_ising(p)
# lp._to_cqm()
#
#
# print("-----------------------------------------------------")
# print("Number of q-bits", lp._count_qubits())
# print("Number of couplings Js:", lp._count_quadratic_couplings())
# print("Number of local filds hs:", lp._count_linear_fields())
# print("-----------------------------------------------------")
#
#
#
# simulation = False
#
# if simulation:
#     sdict={"num_sweeps":5_000, "num_reads":10_000, "beta_range":(0.001, 100)}
#     dict_list = annealing(lp, "sim", "12_AGV", sim_anneal_var_dict=sdict, load=False, store=False)
#     print("Simulated annealing results")
#     print_results(dict_list)
#
#
# dict_list = annealing(lp, "hyb", "12_AGV", load=False, store=True)
# print("QPU results")
# print_results(dict_list)
#

"""
dict_list = annealing(lp, "cqm", "12_AGV", load=False, store=False)
print("CQM results:")
print_results(dict_list)



dict_list = annealing(lp, "real", "12_AGV", load=True, store=False)
print("QPU results")
print_results(dict_list)
"""