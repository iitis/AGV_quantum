# 2 AGV example
from dimod.sampleset import SampleSet
import dimod
import minorminer
from dwave.system import DWaveSampler
from src import utils
from src.linear_solver import solve
from src.linear_solver import make_linear_problem, create_linear_model
from src.linear_solver import print_ILP_size
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
from src.quadratic_solver import quadratic_solve_qubo, check_solution, save_results
from src.utils import check_solution_list

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


if __name__ == "__main__":

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
    #
    #
    # model = create_linear_model(obj, A_ub, b_ub, A_eq, b_eq, bounds, iterators)
    # # model.export_as_lp(basename="tiny",  path=os.getcwd())
    # # model.print_information()
    #
    # # begin = time.time()
    # s = model.solve()
    # # end = time.time()
    # # print("time: ", end-begin)
    # model.print_solution(print_zeros=True)
    #
    # # print(model.solve_details)
    # #
    #
    # # model = utils.load_docpex_model("tiny.lp")
    # # model.print_information()
    # # begin = time.time()
    # # s = model.solve()
    # # end = time.time()
    # # print("time: ", end-begin)
    # # model.print_solution(print_zeros=True)
    # # print(model.solve_details)
    # # #
    # #
    # # QUBO
    lp = LinearProg(c=obj, bounds=bounds, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)
    p = 5
    #
    # with open("lp_files/lp_tiny.pkl", "wb") as f:
    #     pickle.dump(lp, f)
    # with open("lp_tiny.pkl", "wb") as f:
    #     pickle.dump(lp, f)
    lp._to_bqm_qubo_ising(p)





    utils.make_spinglass_ising(lp, "tiny")

    #
    # pegasus = DWaveSampler(solver="Advantage_system6.1").to_networkx_graph()
    #
    # qubo2 = minorminer.find_embedding(qubo, pegasus)
    # print(qubo2)
    # #
    # # with open("tiny_qubo.txt", "w") as f:
    # #     f.write(str(qubo))
    #sol, lp = quadratic_solve_qubo(f"lp_tiny.pkl")
    # # s_l = [int(sol.get_value(var)) for var in lp.bqm.variables]
    # # print(s_l)
    # # print(sol.objective_value)
    # # print(check_solution(sol, lp))
    #var = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1]
    #print(check_solution_list(var, lp))

    #
    #
    # # this is QUBO
    # # with open("qubo_tiny.pkl", "wb") as f:
    # #     pickle.dump(lp.qubo, f)
    #
    #
    #
    # print("-----------------------------------------------------")
    # print("Number of q-bits", lp._count_qubits())
    # print("Number of couplings Js:", lp._count_quadratic_couplings())
    # print("Number of local filds hs:", lp._count_linear_fields())
    #

    # sdict={"num_sweeps":1_000, "num_reads":500, "beta_range":(0.01, 20)}
    # dict_list = annealing(lp, "sim", "2_tiny_AGV", sim_anneal_var_dict=sdict, load=False, store=False)
    # print("Simulated annealing lp_files")
    # print_results(dict_list)

    # sdict={"num_sweeps":1_000, "num_reads":500, "beta_range":(0.01, 20)}
    # dict_list = annealing(lp, "sim", "2_tiny_AGV", sim_anneal_var_dict=sdict, load=False, store=False)
    # print("Simulated annealing lp_files")
    # print_results(dict_list)

    # d1 = lp.bqm.quadratic
    # d2 = lp.bqm.linear
    # max_d1 = abs(d1[max(d1, key=lambda y: abs(d1[y]))])
    # max_d2 = abs(d2[max(d2, key=lambda y: abs(d2[y]))])
    # max_bqm = max(max_d1, max_d2)
    #
    # #int(max_bqm + sqrt(max_bqm))
    # rdict = {"num_reads": 2200, "annealing_time": 250, 'chain_strength': 4, 'solver': 'Advantage_system6.1'}
    # dict_list = annealing(lp, "real", "2_tiny_AGV", load=False, store=True)
    # print("QPU lp_files")
    # print_results(dict_list)

    """
    name = "tiny"
    sol, lp = quadratic_solve_qubo(f"lp_{name}.pkl")
    sol.export(f"sol_{name}.json")
    feasible, lp_files = check_solution(sol, lp)
    save_results(lp_files, f"{name}", "lp_files.txt")
    """


    """
    dict_list = annealing(lp, "cqm", "2_tiny_AGV", load=False, store=False)
    print("CQM lp_files:")
    print_results(dict_list)
    
    
    dict_list = annealing(lp, "hyb", "2_tiny_AGV", load=False, store=True)
    print("QPU lp_files")
    print_results(dict_list)
    
    dict_list = annealing(lp, "real", "2_tiny_AGV", load=True, store=False)
    print("QPU lp_files")
    print_results(dict_list)
    """