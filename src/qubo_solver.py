import os

import neal
import dimod
from dwave.system import (
    EmbeddingComposite,
    DWaveSampler,
    LeapHybridSampler,
    LeapHybridCQMSampler,
)
from scipy.optimize import linprog

from src.LinearProg import LinearProg
from src.process_results import get_results, load_results, print_results, store_result
from typing import Optional

def sim_anneal(
    bqm: dimod.BinaryQuadraticModel,
    beta_range,
    num_sweeps,
    num_reads,
) -> dimod.sampleset.SampleSet:
    """Runs simulated annealing experiment

    :param bqm: binary quadratic model to be sampled
    :type bqm: dimod.BinaryQuadraticModel
    :param beta_range: beta range for the experiment
    :type beta_range: Tuple(int,int)
    :param num_sweeps: Number of steps
    :type num_sweeps: int
    :param num_reads: Number of samples
    :type num_reads: int
    :return: sampleset
    :rtype: dimod.SampleSet
    """
    s = neal.SimulatedAnnealingSampler()
    sampleset = s.sample(
        bqm,
        beta_range=beta_range,
        num_sweeps=num_sweeps,
        num_reads=num_reads,
        beta_schedule_type="geometric",
    )
    return sampleset


def real_anneal(
    bqm: dimod.BinaryQuadraticModel,
    num_reads: int,
    annealing_time: float,
    chain_strength: float,
    solver: Optional[str] = 'Advantage_system6.2'
) -> dimod.sampleset.SampleSet:
    """Runs quantum annealing experiment on D-Wave

    :param bqm: binary quadratic model to be sampled
    :type bqm: dimod.BinaryQuadraticModel
    :param num_reads: Number of samples
    :type num_reads: int
    :param annealing_time: Annealing time
    :type annealing_time: int
    :param chain_strength: Chain strength parameters
    :type chain_strength: float
    :return: sampleset
    :rtype: dimod.SampleSet
    """
    sampler = EmbeddingComposite(DWaveSampler(solver=solver))
    # annealing time in micro second, 20 is default.
    sampleset = sampler.sample(
        bqm,
        num_reads=num_reads,
        auto_scale="true",
        annealing_time=annealing_time,
        chain_strength=chain_strength,
    )
    return sampleset


def constrained_solver(
    cqm: dimod.ConstrainedQuadraticModel,
) -> dimod.sampleset.SampleSet:
    """Runs experiment using constrained solver

    :param cqm: Constrained model for the problem
    :type cqm: dimod.ConstrainedQuadraticModel
    :return: sampleset
    :rtype: dimod.SampleSet
    """
    sampler = LeapHybridCQMSampler()
    return sampler.sample_cqm(cqm)


def hybrid_anneal(bqm: dimod.BinaryQuadraticModel) -> dimod.sampleset.SampleSet:
    """Runs experiment using hybrid solver

    :param bqm: Binary quadratic model for the problem
    :type bqm: dimod.BinaryQuadraticModel
    :return: sampleset
    :rtype: dimod.SampleSet
    """
    sampler = LeapHybridSampler()
    return sampler.sample(bqm)


def get_file_name(
    input_name: str,
    method: str,
    num_reads=None,
    annealing_time=None,
    chain_strength=None,
) -> str:
    """Generates a file name based on the given parameters

    :param input_name: Name of the input data
    :type input_name: str
    :param method: method of execution
    :type method: str
    :param num_reads: Number of reads in QA
    :type num_reads: int
    :param annealing_time: Annealing time for QA
    :type annealing_time: int
    :param chain_strength: Chain strength for QA
    :type chain_strength: int
    :return: name of the file to be stored
    :rtype: str
    """
    folder = os.path.join("annealing_results", input_name)
    fname = f"{method}"
    if num_reads != None:
        fname += f"_{num_reads}_{annealing_time}_{chain_strength}"
    return os.path.join(folder, fname)


def get_parameters(real_anneal_var_dict: dict[str, float]) -> tuple[int, int, int, str]:
    """Extracts/sets parameters for annealing experiment

    :param real_anneal_var_dict: Parameters for QA experiment
    :type real_anneal_var_dict: dict[str, float]
    :return: Number of reads, annealing_time and chain strength
    :rtype: Tuple[int, int, int]
    """
    if real_anneal_var_dict == None:
        num_reads = 1000
        annealing_time = 250
        chain_strength = 4
        solver = 'Advantage_system6.1'
    else:
        num_reads = real_anneal_var_dict["num_reads"]
        annealing_time = real_anneal_var_dict["annealing_time"]
        chain_strength = real_anneal_var_dict["chain_strength"]
        solver = real_anneal_var_dict["solver"] if len(real_anneal_var_dict)>3 else None

    return num_reads, annealing_time, chain_strength, solver


def annealing(
    lp: LinearProg,
    method: str,
    input_name: str,
    real_anneal_var_dict=None,
    sim_anneal_var_dict=None,
    load=False,
    store=True,
) -> list[dict]:
    """Performs the annealing experiment

    :param lp: The linear program instance
    :type lp: LinearProg
    :param method: 'sim', 'real', 'hyb', 'cqm'
    :type method: str
    :param input_name: name of the input data
    :type input_name: str
    :param real_anneal_var_dict: Parameters for QA
    :type real_anneal_var_dict: Dict[str, float]
    :return: list of result dictionary
    :rtype: list(dict)
    """

    assert method in ["sim", "real", "hyb", "cqm"]
    if method == "real":
        num_reads, annealing_time, chain_strength, solver = get_parameters(real_anneal_var_dict)
        file_name = get_file_name(
            input_name,
            method,
            num_reads,
            annealing_time,
            chain_strength,
        )
    else:
        file_name = get_file_name(input_name, method)
        print(file_name)
    if load:
        try:
            sampleset = load_results(file_name)
        except:
            try:
                sampleset = load_results("examples/" + file_name)
            except FileNotFoundError:
                print("File does not exist")
                exit()
    else:
        if method == "cqm":
            cqm = lp.cqm
            sampleset = constrained_solver(cqm)
        else:
            bqm = lp.bqm
            if method == "sim":
                if sim_anneal_var_dict is not None:
                    num_sweeps = sim_anneal_var_dict["num_sweeps"]
                    num_reads = sim_anneal_var_dict["num_reads"]
                    r = sim_anneal_var_dict["beta_range"]
                else:
                    num_sweeps = 1000
                    num_reads = 1000
                    r = (1,100)
                sampleset = sim_anneal(
                    bqm, beta_range=r, num_sweeps=num_sweeps, num_reads=num_reads
                )
            elif method == "real":
                if solver:
                    sampleset = real_anneal(
                        bqm,
                        num_reads=num_reads,
                        annealing_time=annealing_time,
                        chain_strength=chain_strength,
                        solver=solver
                    )
                else:
                    sampleset = real_anneal(
                        bqm,
                        num_reads=num_reads,
                        annealing_time=annealing_time,
                        chain_strength=chain_strength
                    )
            elif method == "hyb":
                sampleset = hybrid_anneal(bqm)
    if store:
        if os.path.exists(file_name):
            print("Overwriting lp_files")
        store_result(input_name, file_name, sampleset)
    if method != "cqm":
        sampleset = lp.interpreter(sampleset)
    return get_results(sampleset, prob=lp)


