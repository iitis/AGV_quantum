from dimod.sampleset import SampleSet
from src.process_results import get_results
from src.quadratic_solver import load_linear_prog_object, process_result
import json
from src.utils import check_solution_list

with open("results/out_large.json", "r") as f:
    res = json.load(f)

lp = load_linear_prog_object("lp_files/lp_large.pkl")


res = res["result"]
print(check_solution_list(res, lp))
