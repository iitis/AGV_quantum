from src.LinearProg import LinearProg
import pickle
import json
from docplex.mp.model import Model
from docplex.mp.solution import SolveSolution
from src.process_results import analyze_constraints, get_results
import dimod
from docplex.mp.progress import ProgressListener, ProgressClock, TextProgressListener
import docplex.util.environment as environment
from typing import Union


class AutomaticAborter(ProgressListener):
    """ a simple implementation of an automatic search stopper.
        WIP
    """

    def __init__(self, max_no_improve_time=10.):
        super(AutomaticAborter, self).__init__(ProgressClock.All)
        self.last_obj = None
        self.last_obj_time = None
        self.max_no_improve_time = max_no_improve_time

    def notify_start(self):
        super(AutomaticAborter, self).notify_start()
        self.last_obj = None
        self.last_obj_time = None

    def is_improving(self, new_obj, eps=1e-4):
        last_obj = self.last_obj
        return last_obj is None or (abs(new_obj - last_obj) >= eps)

    def notify_progress(self, pdata):
        super(AutomaticAborter, self).notify_progress(pdata)
        if pdata.has_incumbent and self.is_improving(pdata.current_objective):
            self.last_obj = pdata.current_objective
            self.last_obj_time = pdata.time
            print('----> #new objective={0}, time={1}s'.format(self.last_obj, self.last_obj_time))
        else:
            # a non improving move
            last_obj_time = self.last_obj_time
            this_time = pdata.time
            if last_obj_time is not None:
                elapsed = (this_time - last_obj_time)
                if elapsed >= self.max_no_improve_time:
                    print('!! aborting cplex, elapsed={0} >= max_no_improve: {1}'.format(elapsed,
                                                                                         self.max_no_improve_time))
                    self.abort()
                else:
                    print('----> non improving time={0}s'.format(elapsed))



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


def quadratic_solve_qubo(lp_location: str, num_threads: int = None) -> (SolveSolution, LinearProg):
    lp = load_linear_prog_object(lp_location)
    lp = add_zero_h_qubo(lp)
    qubo = lp.qubo[0]

    m = Model(name='qubo')
    if num_threads:
        m.context.cplex_parameters.threads = num_threads
    variables = m.binary_var_dict(lp.bqm.variables, name="", key_format="%s")
    obj_fnc = sum(variables[k1] * variables[k2] * qubo[(k1, k2)] for k1, k2 in qubo.keys())
    m.set_objective("min", obj_fnc)
    #m.print_information()
    m.add_progress_listener(TextProgressListener(clock="objective"))
    sol = m.solve()
    #m.print_solution()
    return sol, lp


def quadratic_model(lp_location: str) -> Model:
    """
    :param lp_location: path to lp file
    :return: docplex model for given qubo
    """
    lp = load_linear_prog_object(lp_location)
    lp = add_zero_h_qubo(lp)
    qubo = lp.qubo[0]

    model = Model(name='qubo')
    variables = model.binary_var_dict(lp.bqm.variables, name="", key_format="%s")
    obj_fnc = sum(variables[k1] * variables[k2] * qubo[(k1, k2)] for k1, k2 in qubo.keys())
    model.set_objective("min", obj_fnc)

    return model

def process_result(sampleset: list):
    energy = sampleset[0]["energy"]
    objective = sampleset[0]["objective"]
    feasible = sampleset[0]["feasible"]
    broken_constrains = []
    if not feasible:
        for eq, feas in sampleset[0]["feas_constraints"][0].items():
            if not feas:
                broken_constrains.append(eq)
    num_broken = len(broken_constrains)

    return {"energy": energy, "objective": objective, "feasible": feasible, "broken_constrains": broken_constrains,
            "num_broken": num_broken}


def check_solution(sol: Union[SolveSolution, dict], lp: LinearProg):
    if isinstance(sol, dict):
        raise NotImplementedError
    else:
        solved_list = [v.name for v in sol.iter_variables()]
        sample = {v: 1 if v in solved_list else 0 for v in lp.bqm.variables}
        sampleset = dimod.SampleSet.from_samples(dimod.as_samples(sample), 'BINARY', sol.objective_value)
        sampleset = lp.interpreter(sampleset)
        results = get_results(sampleset, prob=lp)
        results = process_result(results)

        return results["feasible"], results


def save_results(results: dict, name:str, output_path: str):

    with open(output_path, "a") as f:
        f.write(f"{name}: ")
        f.write(str(results))


if __name__ == "__main__":
    for name in ["tiny", "smallest", "small", "medium_small"]:
        sol, lp = quadratic_solve_qubo(f"lp_{name}.pkl", num_threads=18)
        sol.export(f"sol_{name}.json")
        feasible, results = check_solution(sol, lp)
        save_results(results, f"{name}", "results.txt")


