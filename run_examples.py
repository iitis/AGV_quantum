"execute examples"

import pickle
import time
import os

from AGV_quantum import plot_train_diagram
from AGV_quantum import print_ILP_size, LinearAGV
from AGV_quantum import QuadraticAGV
from AGV_quantum import constrained_solver, hybrid_anneal


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
parser.add_argument(
    "--example",
    type=str,
    help="chose example []",
    default="smallest",
)


args = parser.parse_args()
cwd = os.getcwd()
if args.example == "tiny":
    from examples.example_tiny import *
    save_path = os.path.join(cwd, "..", "annealing_results", "tiny_2_AGV")
if args.example == "smallest":
    from examples.example_smallest import *
    save_path = os.path.join(cwd, "..", "annealing_results", "2_AGV")
if args.example == "small":
    from examples.example_small import *
    save_path = os.path.join(cwd, "..", "annealing_results", "4_AGV")
if args.example == "medium_small":
    from examples.example_medium_small import *
    save_path = os.path.join(cwd, "..", "annealing_results", "6_AGV")
if args.example == "medium":
    from examples.example_medium import *
    save_path = os.path.join(cwd, "..", "annealing_results", "7_AGV")
if args.example == "large":
    from examples.example_large import *
    save_path = os.path.join(cwd, "..", "annealing_results", "12_AGV")
if args.example == "largest":
    from examples.example_largest import *
    save_path = os.path.join(cwd, "..", "annealing_results", "15_AGV")
else:
    print(f"example {args.example} not suported")

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
            plot_train_diagram(sol, agv_routes, tracks_len)

    if solve_quadratic:
        ising_size = True
        hybrid = "bqm" # select hybrid solver bqm or cqm
        p = 5 # penalty for QUBO creation

        model = QuadraticAGV(AGV)
        model.to_bqm_qubo_ising(p)
        if ising_size:
            print("n.o. qubits", model._count_qubits())
            print("n.o. quandratic cuplings", model._count_quadratic_couplings())
            print("n.o. linear fields", model._count_linear_fields())
        else:
            model.to_cqm()
            cwd = os.getcwd()
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