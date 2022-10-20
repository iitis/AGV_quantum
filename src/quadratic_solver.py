from src.LinearProg import LinearProg
import pickle
import json
from docplex.mp.model import Model
from docplex.mp.solution import SolveSolution
from src.process_results import analyze_constraints, get_results
import dimod
import docplex.mp.progress as prog
from typing import Union

def add_zero_h_qubo(lp: LinearProg) -> LinearProg:
    for var in lp.bqm.variables:
        if (var, var) not in lp.qubo[0]:
            lp.qubo[0][(var, var)] = 0
    return lp


def add_zero_h_ising(lp: LinearProg) -> LinearProg:
    for var in lp.bqm.variables:
        if var not in lp.ising[0]:
            lp.ising[0][var] = 0
    return lp


def load_linear_prog_object(lp_location: str) -> LinearProg:
    with open(lp_location, "rb") as f:
        lp = pickle.load(f)
    p = 2.75
    lp._to_bqm_qubo_ising(p)
    return lp

def count_vertices(qubo: dict) -> int:
    s = 0
    for key in qubo.keys():
        if key[0] == key[1]:
            s += 1
    return s


def count_edges(qubo: dict) -> int:
    s = 0
    for key in qubo.keys():
        if key[0] != key[1]:
            s += 1
    return s


def quadratic_solve_qubo(lp_location: str):
    lp = load_linear_prog_object(lp_location)
    lp = add_zero_h_qubo(lp)
    qubo = lp.qubo[0]

    m = Model(name='qubo')
    variables = m.binary_var_dict(lp.bqm.variables, name="", key_format="%s")
    obj_fnc = sum(variables[k1] * variables[k2] * qubo[(k1, k2)] for k1, k2 in qubo.keys())
    m.set_objective("min", obj_fnc)
    m.print_information()
    m.add_progress_listener(prog.TextProgressListener())
    sol = m.solve()
    m.print_solution()
    return sol, lp


def check_solution(sol: Union[SolveSolution, dict], lp: LinearProg):

    if isinstance(sol, dict):
        raise NotImplementedError
    else:
        solved_list = [v.name for v in sol.iter_variables()]
        sample = {v: 1 if v in solved_list else 0 for v in lp.bqm.variables}
        sampleset = dimod.SampleSet.from_samples(dimod.as_samples(sample), 'BINARY', sol.objective_value)
        sampleset = lp.interpreter(sampleset)
        print(get_results(sampleset, prob=lp))

if __name__ == "__main__":
    for name in ["tiny, smallest, small, medium_small"]:
        sol, lp = quadratic_solve_qubo(f"../lp_{name}.pkl")
        sol.export(f"../sol_{name}.json")
        check_solution(sol, lp)



