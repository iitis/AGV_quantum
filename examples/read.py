from src.process_results import load_results, print_results, get_results, get_objective
from src.utils import print_time, print_best_feasible
from dwave.system import LeapHybridSampler
from dimod.serialization.format import Formatter
from dwave.cloud.client import Client
from src.LinearProg import LinearProg
import pickle

def load_linear_prog_object(lp_location: str) -> LinearProg:
    with open(lp_location, "rb") as f:
        lp = pickle.load(f)
    p = 2.75
    lp._to_bqm_qubo_ising(p)
    return lp

#client = Client.from_config()


sampler = LeapHybridSampler()
lp = load_linear_prog_object("lp_smallest.pkl")
re = load_results("annealing_results/2_AGV/real_2200_250_1765")
print(re.first)
print(re.info['timing'])
#print_time(re)
#re_dict = get_results(re, lp)
#print_best_feasible(re)

