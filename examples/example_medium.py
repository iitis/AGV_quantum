from src import utils
from src.linear_solver import solve
from src.linear_solver import make_linear_problem, create_linear_model
from src import train_diagram
from src.qubo_solver import annealing
from src.linear_solver import print_ILP_size
import numpy as np
import pickle
import time
import os

from src.LinearProg import LinearProg
from src.process_results import print_results

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
              1: ("s0", "s1", "s2"),
              2: ("s4", "s3", "s2", "s1"),
              3: ("s4", "s3", "s2", "s1", "s0"),
              4: ("s2", "s3"),
              5: ("s6", "s5", "s4", "s3"),
              6: ("s5", "s6")}

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

initial_conditions = {("in", 0, "s0"): 0, ("in", 1, "s0"): 0, ("in", 2, "s4"): 8, ("in", 3, "s4"): 9,
                      ("in", 4, "s2"): 15, ("in", 5, "s6"): 0, ("in", 6, "s5"): 0}

weights = {j: 1 for j in J}

obj, A_ub, b_ub, A_eq, b_eq, bounds, iterators = make_linear_problem(M, tracks, tracks_len, agv_routes, d_max,
                                                 tau_pass, tau_headway, tau_operation, weights, initial_conditions)

res, iterators = solve(obj, A_ub, b_ub, A_eq, b_eq, bounds, iterators)
#these are paths to train diagram plot
complete_path = {"s0_in":0,"s0_out":2,"s1_in":8,"s1_out":10,"s2_in":16, "s2_out":18,"s3_in":18,"s3_out":20,"s4_in":25, "s4_out":27, "s5_in":31, "s5_out":33, "s6_in":37, "s6_out":39}
complete_path_rev = {"s0_out":0,"s0_in":2,"s1_out":8,"s1_in":10,"s2_out":16, "s2_in":18,"s3_out":18,"s3_in":20,"s4_out":25, "s4_in":27, "s5_out":31, "s5_in":33, "s6_out":37, "s6_in":39}
path_locs = [0,2,8,10,16, 18,18,20,25, 27, 31, 33, 37, 39]

# linear solver
if res.success:
    v_in, v_out = utils.create_v_in_out(tracks_len, agv_routes, tau_operation, iterators, initial_conditions)
    utils.nice_print(res, agv_routes, weights, d_max,  v_in, v_out, iterators)
    times, paths = utils.get_data4plot(res, agv_routes, iterators, complete_path, complete_path_rev, rev = [2,3,5])
    train_diagram.plot_train_diagram(times, paths, path_locs)
    
else:
    print(res.message)

print_ILP_size(A_ub, b_ub, A_eq, b_eq)

model = create_linear_model(obj, A_ub, b_ub, A_eq, b_eq, bounds, iterators)
model.print_information()
begin = time.time()
s = model.solve()
end = time.time()
print("time: ", end-begin)
model.print_solution(print_zeros=True)
print(model.solve_details)

lp = LinearProg(c=obj, bounds=bounds, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)

p = 2.75

lp._to_bqm_qubo_ising(p)

qubo = lp.qubo[0]
qubo = dict(sorted(qubo.items()))
lin = lp.ising[0]
quad = lp.ising[1]
quad = dict(sorted(quad.items()))
ising_offset = lp.ising[2]

def make_spin_glass_qubo():
    def number():
        i = 1
        while True:
            yield i
            i += 1

    number = number()

    linear_qubo = []
    for key1, key2 in qubo.keys():
        if key1 == key2:
            linear_qubo.append(key1)

    key_numbers = {key: next(number) for key in linear_qubo}
    spinglass_qubo = {}
    for (key1, key2), value in qubo.items():
        spinglass_qubo[(key_numbers[key1], key_numbers[key2])] = value

    print(spinglass_qubo)

    with open(os.path.join(cwd, "..", "qubo", "tiny_qubo_spinglass.txt"), "w") as f:
        f.write(f"# offset: {lp.qubo[1]} \n")
        for (i, j), v in spinglass_qubo.items():
            if i == j:
                f.write(f"{i} {j} {v} \n")
        for (i, j), v in spinglass_qubo.items():
            if i != j:
                f.write(f"{i} {j} {v} \n")


def make_ising_spinglass():
    number = utils.number()
    set_of_keys = set()
    for i, j in quad.keys():
        set_of_keys.add(i)
        set_of_keys.add(j)
    keys_number = {key: next(number) for key in set_of_keys}

    with open(os.path.join(cwd, "..", "qubo", "medium_ising_spinglass.txt"), "w") as f:
        f.write(f"# offset: {ising_offset} \n")
        for key in set_of_keys:
            num = keys_number[key]
            v = lin[key] if key in lin.keys() else 0
            f.write(f"{num} {num} {v} \n")
        for (i, j), v in quad.items():
            ni = keys_number[i]
            nj = keys_number[j]
            f.write(f"{ni} {nj} {v} \n")

make_ising_spinglass()


# QUBO
#
# lp = LinearProg(c=obj, bounds=bounds, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)
# p = 2.75
#
# with open("lp_medium.pkl", "wb") as f:
#     pickle.dump(lp, f)
#
# lp._to_bqm_qubo_ising(p)
# lp._to_cqm()
#
#
#
# print("-----------------------------------------------------")
# print("Number of q-bits", lp._count_qubits())
# print("Number of couplings Js:", lp._count_quadratic_couplings())
# print("Number of local filds hs:", lp._count_linear_fields())
# print("-----------------------------------------------------")
#
# simulation = False
#
# if simulation:
#     sdict={"num_sweeps":10_000, "num_reads":1_000, "beta_range":(0.0001, 100)}
#     dict_list = annealing(lp, "sim", "7_AGV", sim_anneal_var_dict=sdict, load=False, store=False)
#     print("Simulated annealing lp_files")
#     print_results(dict_list)
#
# dict_list = annealing(lp, "hyb", "7_AGV", load=False, store=True)
# print("QPU lp_files")
# print_results(dict_list)
#
#

"""




dict_list = annealing(lp, "cqm", "7_AGV", load=True, store=False)
print("CQM lp_files:")
print_results(dict_list)

dict_list = annealing(lp, "real", "7_AGV", load=True, store=False)
print("QPU lp_files")
print_results(dict_list)
"""