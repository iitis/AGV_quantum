"execute examples"

import pickle
import time
import os
import argparse

from AGV_quantum import plot_train_diagram
from AGV_quantum import print_ILP_size, LinearAGV
from AGV_quantum import QuadraticAGV
from AGV_quantum import constrained_solver, hybrid_anneal


parser = argparse.ArgumentParser("Solve linear or quadratic")
parser.add_argument(
    "--solve_linear",
    type=int,
    help="1 if solve the linear problem on CPLEX, 0 is solve the problem on hybrid quantum approach",
    default=1,
)
parser.add_argument(
    "--train_diagram",
    type=int,
    help="Make train diagram for linear solution",
    default=0,
)
parser.add_argument(
    "--example",
    type=str,
    help="chose example out of [tiny, smallest, small, medium_small, medium, large, largest]",
    default="smallest",
)
parser.add_argument(
    "--hyb_solver",
    type=str,
    help="chose bqm or cqm",
    default="cqm",
)

count = "_10"
count = ""

args = parser.parse_args()
cwd = os.getcwd()
if args.example == "tiny":
    from examples.example_tiny import M, tracks, tracks_len, agv_routes, d_max, tau_pass, tau_headway, tau_operation, weights, initial_conditions
    save_path = os.path.join(cwd, "annealing_results", "tiny_2_AGV")
elif args.example == "smallest":
    from examples.example_smallest import M, tracks, tracks_len, agv_routes, d_max, tau_pass, tau_headway, tau_operation, weights, initial_conditions
    save_path = os.path.join(cwd, "annealing_results", "2_AGV")
elif args.example == "small":
    from examples.example_small import M, tracks, tracks_len, agv_routes, d_max, tau_pass, tau_headway, tau_operation, weights, initial_conditions
    save_path = os.path.join(cwd, "annealing_results", "4_AGV")
elif args.example == "medium_small":
    from examples.example_medium_small import M, tracks, tracks_len, agv_routes, d_max, tau_pass, tau_headway, tau_operation, weights, initial_conditions
    save_path = os.path.join(cwd, "annealing_results", "6_AGV")
elif args.example == "medium":
    from examples.example_medium import M, tracks, tracks_len, agv_routes, d_max, tau_pass, tau_headway, tau_operation, weights, initial_conditions
    save_path = os.path.join(cwd, "annealing_results", "7_AGV")
elif args.example == "large":
    from examples.example_large import M, tracks, tracks_len, agv_routes, d_max, tau_pass, tau_headway, tau_operation, weights, initial_conditions
    save_path = os.path.join(cwd, "annealing_results", "12_AGV")
elif args.example == "largest":
    from examples.example_largest import M, tracks, tracks_len, agv_routes, d_max, tau_pass, tau_headway, tau_operation, weights, initial_conditions
    save_path = os.path.join(cwd, "annealing_results", "15_AGV")
elif args.example == "largest_ever":
    from examples.example_largest_ever import M, tracks, tracks_len, agv_routes, d_max, tau_pass, tau_headway, tau_operation, weights, initial_conditions
    save_path = os.path.join(cwd, "annealing_results", "21_AGV")
else:
    print(f"example {args.example} not suported")

solve_linear = args.solve_linear


if __name__ == "__main__":

    not_qubo_method = ""

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
        #AGV.nice_print(model, sol) <- WIP
        if args.train_diagram:
            plot_train_diagram(sol, agv_routes, tracks_len, "CPLEX")

    else:

        assert args.hyb_solver in ["bqm", "cqm"]

        hybrid = args.hyb_solver
        p = 5 # penalty for QUBO creation
        model = QuadraticAGV(AGV)

        # saves model for checks
        lp_file = os.path.join(cwd, f"lp_files/lp_{args.example}.pkl")
        if not os.path.isfile(lp_file):
            with open(lp_file, "wb") as f:
                pickle.dump(model, f)

        model.to_bqm_qubo_ising(p)

        print("n.o. qubits", model._count_qubits())
        print("n.o. quandratic couplings", model._count_quadratic_couplings())
        print("n.o. linear fields", model._count_linear_fields())

        # check if results are saved
        is_file = os.path.isfile(os.path.join(save_path, f"new{not_qubo_method}_{hybrid}_info{count}.pkl"))
        if is_file:
            print(".......... files exist ............")

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

            with open(os.path.join(save_path, f"new{not_qubo_method}_{hybrid}_info{count}.pkl"), "wb") as f:
                pickle.dump(info, f)

            with open(os.path.join(save_path, f"new{not_qubo_method}_{hybrid}{count}.pkl"), "wb") as f:
                pickle.dump(sampleset.to_serializable(), f)
