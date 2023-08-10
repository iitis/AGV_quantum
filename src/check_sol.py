import pickle
import os
import dimod
import pyqubo
from src.process_results import get_results, analyze_constraints, get_objective
from src.quadratic_solver_CPLEX import load_linear_prog_object, process_result
import json
from src.utils import check_solution_list
import pandas as pd

from pathlib import Path

cwd = os.getcwd()
sol_folder = Path("annealing_results/7_AGV")
lp_folder = Path("lp_files")

hybrid = "cqm"

with open(os.path.join(cwd, "..", sol_folder, f"new_{hybrid}.pkl"), "rb") as f:
    sampleset = pickle.load(f)

with open(os.path.join(cwd, "..", lp_folder, "lp_medium.pkl"), "rb") as f:
    lp = pickle.load(f)

sampleset = dimod.SampleSet.from_serializable(sampleset)

if __name__ == '__main__':

    if hybrid == "bqm":
        print(sampleset.info)
        p=5
        lp._to_bqm_qubo_ising(p)
        sampleset = lp.interpreter(sampleset)
        for sol in get_results(sampleset, lp):
            print(sol)

    elif hybrid == "cqm":
        print(sampleset.info)
        for sol in get_results(sampleset, lp):
            print(sol)
