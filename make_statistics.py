import pickle
import numpy as np
import os
from pathlib import Path
import dimod
from AGV_quantum import get_results



n_vars = {2:16, 4:36, }


def read_single_file(size, k, solver):

    file = f"annealing_results/{size}_AGV/new_{solver}_{k}.pkl"

    with open (file, "rb") as f:
        d = pickle.load(f)
    

    return d

    


def feasibility_perc_cqm(examples, size, k):

    cwd = os.getcwd()

    lp_folder = Path(f"lp_files/lp_{examples[size]}.pkl")

    with open(os.path.join(cwd, lp_folder), "rb") as f:
        lp = pickle.load(f)

    sampleset = read_smapleset(size, k, "cqm")

    solutions = get_results(sampleset, lp)

    return sum(sol["feasible"] for sol in solutions)/len(solutions)


def objective_cqm(examples, size, k):

    cwd = os.getcwd()

    lp_folder = Path(f"lp_files/lp_{examples[size]}.pkl")


    with open(os.path.join(cwd, lp_folder), "rb") as f:
        lp = pickle.load(f)

    sampleset = read_smapleset(size, k, "cqm")

    solutions = get_results(sampleset, lp)
    solutions = sorted(solutions, key=lambda d: d["feasible"], reverse=True)

    return solutions[0]["objective"]


def obj_cqm_array(examples, size):
    return np.array([objective_cqm(examples, size, k) for k in range(1,11)])


def feas_cqm_array(examples, size):
    return np.array([feasibility_perc_cqm(examples, size, k) for k in range(1,11)])



def read_series(size, solver, key):
    return np.array([read_smapleset(size, k, solver).info[key] for k in range(1,11)])


def read_smapleset(size, k, solver):
    sampleset = read_single_file(size, k, solver)
    sampleset = dimod.SampleSet.from_serializable(sampleset)

    return sampleset


def print_info(key, sizes):
    print(key)
    for size in sizes:
        vars = get_no_vars(size)
        series = read_series(size, "cqm", key)
        if key == "qpu_access_time":
            k = 1000
        elif key == "run_time":
            k=1000000
        else:
            k = 1
        
        print(vars, np.mean(series/k), np.std(series/k))



def get_no_vars(size):

    vars = read_smapleset(size, 1, "cqm").variables
    return (len(vars))


if __name__ == "__main__":

    #sizes = [2,4,6,7,12,15,21]

    optimum = {2:4, 4: 8.2, 6: 3.22, 7: 4.25, 12: 9.175,  15: 10.975 }

    examples = {2:"smallest", 4:"small", 6:"medium_small", 7:"medium", 12:"large", 15:"largest", 21:"largest_ever"}
    sizes = examples.keys()



    for size in sizes:
        obj = obj_cqm_array(examples, size)
        obj = obj/optimum[size]
        
        print(obj)
        print(np.mean(obj), np.std(obj))


    for size in sizes:
        perc = feas_cqm_array(examples, size)
        print(np.mean(perc), np.std(perc))


    key = 'qpu_access_time'
    print_info(key, sizes)

    key = 'run_time'
    print_info(key, sizes)



    