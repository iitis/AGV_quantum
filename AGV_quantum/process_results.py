"""analyse results given quantum/hybrid/simulation approach"""

import os
import pickle
from dimod import SampleSet
from AGV_quantum import LinearProg


def get_objective(lp: LinearProg, sample: dict) -> float:
    """computes objective value for sample

    :param lp: the integer program with the relevant objective function
    :type lp: LinearProg
    :param sample: analyzed sample
    :type sample: dict
    :return: value of the objective funtion
    :rtype: float
    """
    return sum(
        sample[f"x_{i}"] * coef for i, coef in zip(range(lp.nvars), lp.c) if coef != 0
    )


def get_results(sampleset: SampleSet, prob):
    """Check samples one by one, and computes it statistics.

    Statistics includes energy (as provided by D'Wave), objective function
    value, feasibility analysis, the samples itself. Samples are sorted
    according to value of the objetive function

    :param sampleset: analyzed samples
    :type sampleset: dimod.SampleSet
    :param prob: integer problem according to which samples are analyzed
    :type prob: pulp.LpProblem
    :return: analyzed samples, sorted according to objective
    :rtype: list[Dict[str,Any]]
    """
    dict_list = []
    for data in sampleset.data():
        rdict = {}
        sample = data.sample
        rdict["energy"] = data.energy
        rdict["objective"] = round(get_objective(prob, sample), 2)
        rdict["feasible"] = all(analyze_constraints(prob, sample)[0].values())
        rdict["sample"] = sample
        rdict["feas_constraints"] = analyze_constraints(prob, sample)
        dict_list.append(rdict)
    return sorted(dict_list, key=lambda d: d["energy"])


def store_result(input_name: str, file_name: str, sampleset: SampleSet):
    """Save samples to the file

    :param input_name: name of the input
    :type input_name: str
    :param file_name: name of the file
    :type file_name: str
    :param sampleset: samples
    :type sampleset: dimod.SampleSet
    """
    if not os.path.exists("annealing_results"):
        os.mkdir("annealing_results")
    folder = os.path.join("annealing_results", input_name)
    if not os.path.exists(folder):
        os.mkdir(folder)
    sdf = sampleset.to_serializable()
    with open(file_name, "wb") as handle:
        pickle.dump(sdf, handle)


def load_results(file_name: str):
    """Load samples from the file

    :param file_name: name of the file
    :type file_name: str
    :return: loaded samples
    :rtype: dimod.SampleSet
    """
    file = pickle.load(open(file_name, "rb"))
    return SampleSet.from_serializable(file)


def analyze_constraints(lp, sample):
    """check which constraints were satisfied

    :param lp: analyzed integer model
    :type lp: LinearProg
    :param sample: samples generated by the optimizer
    :type sample: Dict[str,int]
    :return: dictionary mapping constraint to whether they were satisfied, and
    the number of satisfied constraints
    :rtype: tuple[dict[str, bool], int]
    """
    result = {}
    num_eq = 0

    if lp.A_eq is not None:
        for i in range(len(lp.A_eq)):
            expr = sum(lp.A_eq[i][j] * sample[lp.var_names[j]] for j in range(lp.nvars))
            result[f"eq_{num_eq}"] = expr == lp.b_eq[i]
            num_eq += 1

    if lp.A_ub is not None:
        for i in range(len(lp.A_ub)):
            expr = sum(lp.A_ub[i][j] * sample[lp.var_names[j]] for j in range(lp.nvars))
            result[f"eq_{num_eq}"] = expr <= lp.b_ub[i]
            num_eq += 1

    not_feas = list(result.values()).count(False)
    return result, not_feas


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

    return {"energy": energy, "objective": objective, "feasible": feasible, "num_broken": num_broken}
