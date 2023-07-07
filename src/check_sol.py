import pickle

from dimod.sampleset import SampleSet
from src.process_results import get_results
from src.quadratic_solver import load_linear_prog_object, process_result
import json
from src.utils import check_solution_list
import pandas as pd
import os

cwd = os.getcwd()

# with open("results/out_large.json", "r") as f:
#     res = json.load(f)
#
# lp = load_linear_prog_object("lp_files/lp_large.pkl")
#
#
# res = res["result"]
# print(check_solution_list(res, lp))

df = pd.read_csv(os.path.join(cwd, "..", "results", "results_sb.csv"), sep=";")


def create_results_dict(df):
    results_dict = {}
    for row in df.itertuples():
        results_dict[row.instance[0:-4]] = list(eval(row.ising_state))
    return results_dict


def read_lp(name):
    with open(os.path.join(cwd, "..", f"lp_{name}.pkl"), "rb") as f:
        lp = pickle.load(f)
        return lp


if __name__ == "__main__":
    name = "large"
    results_dict = create_results_dict(df)
    lp = load_linear_prog_object(os.path.join(cwd, "..", "lp_files", f"lp_{name}.pkl"))
    spins = results_dict[name]
    print(check_solution_list(spins, lp))
