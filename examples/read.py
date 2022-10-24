from src.process_results import load_results, print_results, get_results, get_objective
from src.utils import print_time, print_best_feasible
from dwave.system import LeapHybridSampler
from dimod.serialization.format import Formatter
from dwave.cloud.client import Client
from src.quadratic_solver import load_linear_prog_object


#client = Client.from_config()


sampler = LeapHybridSampler()
lp = load_linear_prog_object("lp_small.pkl")
re = load_results("annealing_results/7_AGV/hyb")
print(re)
print_time(re)
#re_dict = get_results(re, lp)
#print_best_feasible(re)

