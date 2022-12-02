from dimod.sampleset import SampleSet
from src.process_results import get_results, print_results, analyze_constraints, get_objective
from src.quadratic_solver import load_linear_prog_object


lp = load_linear_prog_object("../lp_tiny.pkl")
sol = [0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
sample = SampleSet.from_samples(sol, "BINARY", 0)

print(lp.interpreter(sample))
