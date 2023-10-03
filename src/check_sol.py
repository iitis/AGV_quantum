import pickle
import os
import dimod
#import pyqubo
from src.process_results import get_results
#from src.quadratic_solver_CPLEX import load_linear_prog_object, process_result
#import json
from src.utils import check_solution_list
#import pandas as pd

from pathlib import Path

cwd = os.getcwd()
sol_folder = Path("annealing_results/15_AGV")
lp_folder = Path("lp_files")

hybrid = "bqm"

with open(os.path.join(cwd, "..", sol_folder, f"new_{hybrid}.pkl"), "rb") as f:
    sampleset = pickle.load(f)

with open(os.path.join(cwd, "..", lp_folder, "lp_largest.pkl"), "rb") as f:
    lp = pickle.load(f)

sampleset = dimod.SampleSet.from_serializable(sampleset)

if __name__ == '__main__':

    if hybrid == "bqm":
        print(sampleset.info)
        p=5
        lp._to_bqm_qubo_ising(p)
        sampleset = lp.interpreter(sampleset)
        solutions = get_results(sampleset, lp)
        print(solutions)


    elif hybrid == "cqm":
        print(sampleset.info)
        solutions = get_results(sampleset, lp)
        for sol in solutions:
            if sol["feasible"]:
                print(sol)
