import pickle
import numpy as np
import os
import csv

from pathlib import Path
import dimod
from AGV_quantum import get_results




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
    no_vars = []
    means = []
    stds = []
    for size in sizes:
        vars = get_no_vars(size)
        series = read_series(size, "cqm", key)
        if key == "qpu_access_time":
            k = 1000
        elif key == "run_time":
            k=1000000
        else:
            k = 1
        m = np.mean(series/k)
        s = np.std(series/k)

        no_vars.append(vars)
        means.append(m)
        stds.append(s)
        

        print(vars, m, m-s, m+s)

    return no_vars, means, stds


def print_obj(sizes, examples, optimum):

    no_vars = []
    means = []
    stds = []

    print("objective")
    for size in sizes:
        obj = obj_cqm_array(examples, size)
        obj = obj/optimum[size]
        vars = get_no_vars(size)

        m = np.mean(obj)
        s = np.std(obj)
        print(vars, m,m-s, m+s)


        no_vars.append(vars)
        means.append(m)
        stds.append(s)
        
        print(obj)
        print(np.mean(obj), np.std(obj))

    return no_vars, means, stds


def print_feas(sizes, examples):

    no_vars = []
    means = []
    stds = []

    print("feasibility perc")
    for size in sizes:
        vars = get_no_vars(size)
        perc = feas_cqm_array(examples, size)

        m = np.mean(perc)
        s = np.std(perc)
        print(vars, m,m-s, m+s)


        no_vars.append(vars)
        means.append(m)
        stds.append(s)

    
    return no_vars, means, stds




def csv_write(file_name, no_vars, means, stds):
    with open(file_name, 'w', newline='', encoding="utf-8") as csvfile:
        fieldnames = ["no_vars", "mean", "mean-std", "mean+std"]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        for i,v in enumerate(no_vars):
            m = means[i]
            s = stds[i]
            writer.writerow({"no_vars": v, "mean": np.round(m, 4), "mean-std": np.round(m-s, 4), "mean+std": np.round(m+s, 4)})



def get_no_vars(size):

    vars = read_smapleset(size, 1, "cqm").variables
    return (len(vars))


if __name__ == "__main__":

    #sizes = [2,4,6,7,12,15,21]

    optimum = {2:4, 4: 8.2, 6: 3.22, 7: 4.25, 12: 9.175,  15: 10.975, 21:  17.625}

    examples = {2:"smallest", 4:"small", 6:"medium_small", 7:"medium", 12:"large", 15:"largest", 21:"largest_ever"}
    sizes = examples.keys()


    key = 'qpu_access_time'
    no_vars, means, stds = print_info(key, sizes)

    file = "article_plots/CQM_QPU_time.csv"
    csv_write(file, no_vars, means, stds)

    key = 'run_time'
    no_vars, means, stds = print_info(key, sizes)

    file = "article_plots/time_CQM.csv"
    csv_write(file, no_vars, means, stds)

    no_vars, means, stds = print_feas(sizes, examples)

    file = "article_plots/feasibility_CQM.csv"
    csv_write(file, no_vars, means, stds)

    sizes = examples.keys()

    no_vars, means, stds = print_obj(sizes, examples, optimum)

    file = "article_plots/obj_CQM.csv"
    csv_write(file, no_vars, means, stds)












    