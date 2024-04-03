import pickle
import os
import dimod
from AGV_quantum import get_results, LinearProg


from pathlib import Path

import argparse
parser = argparse.ArgumentParser("Solve linear or quadratic") 
parser.add_argument(
    "--example",
    type=str,
    help="chose example out of [tiny, smallest, small, medium_small, medium, large, largest]",
    default="small",
)
parser.add_argument(
    "--hyb_solver",
    type=str,
    help="chose bqm or cqm",
    default="bqm",
)

args = parser.parse_args()
cwd = os.getcwd()

if args.example == "tiny":
    sol_folder = Path("annealing_results/tiny_2_AGV")
if args.example == "smallest":
    sol_folder = Path("annealing_results/2_AGV")
if args.example == "small":
    sol_folder = Path("annealing_results/4_AGV")
if args.example == "medium_small":
    sol_folder = Path("annealing_results/6_AGV")
if args.example == "medium":
    sol_folder = Path("annealing_results/7_AGV")
if args.example == "large":
    sol_folder = Path("annealing_results/12_AGV")
if args.example == "largest":
    sol_folder = Path("annealing_results/15_AGV")

lp_folder = Path(f"lp_files/lp_{args.example}.pkl")

assert args.hyb_solver in ["bqm", "cqm"]

hybrid = args.hyb_solver

with open(os.path.join(cwd, sol_folder, f"new_{hybrid}.pkl"), "rb") as f:
    sampleset = pickle.load(f)

with open(os.path.join(cwd, lp_folder), "rb") as f:
    lp = pickle.load(f)

sampleset = dimod.SampleSet.from_serializable(sampleset)

if __name__ == '__main__':

    if hybrid == "bqm":
        print(sampleset.info)
        p=5
        lp.to_bqm_qubo_ising(p)
        sampleset = lp.interpreter(sampleset)
        solutions = get_results(sampleset, lp)
        print(solutions)


    elif hybrid == "cqm":
        print(sampleset.info)
        solutions = get_results(sampleset, lp)
        k = 0
        for sol in solutions:
            if sol["feasible"]:
                k = k+1
                #print(sol['objective'])
                if k == 1:
                    print(sol)
        print("no solutions", len(solutions))
        print("feasibility percentage", k/len(solutions))
