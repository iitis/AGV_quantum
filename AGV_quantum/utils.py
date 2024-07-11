#helpers
import itertools
import os
import networkx as nx
import pandas as pd
import numpy as np
import dimod
import pickle

from AGV_quantum import LinearProg
from AGV_quantum import get_results
from AGV_quantum import process_result
from typing import Optional
cwd = os.getcwd()


def create_stations_list(tracks):
    """return list of zones"""
    stations = []
    for track in tracks:
        for station in track:
            stations.append(station)

    return list(set(stations))


def agv_routes_as_edges(agv_routes):
    """ """
    return_dict = {}
    for j in agv_routes.keys():
        if len(agv_routes[j]) > 1:
            s_sp = [(agv_routes[j][i], agv_routes[j][i + 1]) for i in range(len(agv_routes[j]) - 1)]
            return_dict[j] = s_sp
    return return_dict


def create_agv_list(agv_routes):
    return list(agv_routes.keys())


def create_same_way_dict(agv_routes):
    return_dict = {}

    J = agv_routes.keys()
    agv_routes_as_edges_dict = agv_routes_as_edges(agv_routes)

    for j, jp in list(itertools.permutations(J, r=2)):
        temp = []
        for (s, sp) in agv_routes_as_edges_dict[j]:

            if (s, sp) in agv_routes_as_edges_dict[jp]:
                temp.append((s, sp))
        if len(temp) > 0:
            return_dict[(j, jp)] = temp

    return return_dict


def create_graph(tracks, agv_routes):
    stations = create_stations_list(tracks)

    graph = nx.MultiGraph()
    graph.add_nodes_from(stations)
    for track in tracks:
        if len(track) > 1:
            graph.add_edge(track[0], track[1])

    pass_through = {}
    for station in stations:
        temp = []
        for j in agv_routes.keys():
            if station in agv_routes[j]:
                temp.append(j)
        pass_through[station] = temp

    nx.set_node_attributes(graph, pass_through, "pass_through")

    return graph


def create_t_iterator(agv_routes, in_out: str) -> list:
    t_iter = []  # list(itertools.product(J, stations))

    for j, route in agv_routes.items():
        for station in route:
            t_iter.append((in_out, j, station))
    return t_iter


def create_y_iterator(graph: nx.Graph):
    y_iter = []
    agv_pass_through = nx.get_node_attributes(graph, "pass_through")
    for station in sorted(graph.nodes):
        if len(agv_pass_through[station]) >= 2:
            for pair in list(itertools.permutations(agv_pass_through[station], r=2)):
                y_iter.append((pair[0], pair[1], station))
    return y_iter


def create_z_iterator(graph: nx.Graph, agv_routes):
    z_iter = []
    J = create_agv_list(agv_routes)
    agv_routes_as_edges_dict = agv_routes_as_edges(agv_routes)

    for j1, j2 in list(itertools.permutations(J, r=2)):
        if j1 in agv_routes_as_edges_dict and j2 in agv_routes_as_edges_dict:
            for s, sp in agv_routes_as_edges_dict[j1]:
                if graph.number_of_edges(s, sp) < 2 and (sp, s) in agv_routes_as_edges_dict[j2]:
                    z_iter.append((j1, j2, s, sp))

    return z_iter


def create_iterators(graph: nx.Graph, agv_routes):

    t_in_iter = create_t_iterator(agv_routes,  "in")
    t_out_iter = create_t_iterator(agv_routes, "out")
    t_iter = t_in_iter + t_out_iter

    y_iter = create_y_iterator(graph)

    z_iter = create_z_iterator(graph, agv_routes)

    x_iter = t_iter + y_iter + z_iter

    return {"t_in": t_in_iter, "t_out": t_out_iter, "t": t_iter, "y": y_iter, "z": z_iter, "x": x_iter}


def see_variables(vect: list, x_iter: list) -> dict:
    return {x_iter[i]: vect[i] for i in range(len(x_iter))}


def see_non_zero_variables(vect: list, x_iter: list) -> dict:
    return_dict = {}
    for i, _ in enumerate(x_iter):
        if vect[i] != 0:
            return_dict[x_iter[i]] = vect[i]

    return return_dict



def create_v_in_out(tracks_len: dict, agv_routes: dict, tau_operation: dict, iterators: dict, initial_conditions: dict):
    t_in_iter = iterators["t"]

    if not tracks_len:
        return {(j, s): initial_conditions[("in", j, s)] for _, j, s in t_in_iter}, \
               {(j, s): initial_conditions[("in", j, s)] + tau_operation[j,s] for _, j, s in t_in_iter}

    J = create_agv_list(agv_routes)
    v = {}

    for j in J:
        for i, s in enumerate(agv_routes[j]):
            if i == 0:
                v[(j, s)] = initial_conditions[("in", j, s)]
            else:
                s_before = agv_routes[j][i-1]
                v[(j, s)] = v[(j, agv_routes[j][i-1])] + tau_operation[(j, s_before)] + tracks_len[(s_before, s)]

    v_in = v
    v_out = {(j, s): v_in[(j, s)] + tau_operation[(j, s)] for _, j, s in t_in_iter}
    return v_in, v_out



def qubo_to_matrix(qubo: dict, lp: LinearProg) -> np.ndarray:
    qubo = dict(sorted(qubo.items()))
    data = sorted(list(lp.bqm.variables))

    df = pd.DataFrame(columns=data, index=data)
    for item, value in qubo.items():
        df.at[item[0], item[1]] = value
        df.at[item[1], item[0]] = value
    df.fillna(0, inplace=True)
    array = df.to_numpy()

    array = np.triu(array)
    return array


def check_solution_list(sol: list, lp: LinearProg):
    data = sorted(list(lp.bqm.variables))
    sol_dict = {data[i]: sol[i] for i in range(len(sol))}
    offset = lp.qubo[1]
    print(offset)
    #matrix = qubo_to_matrix(lp.qubo[0], lp)
    energy = compute_energy(sol, lp)
    sampleset = dimod.SampleSet.from_samples(dimod.as_samples(sol_dict), 'BINARY', energy)
    sampleset = lp.interpreter(sampleset)
    print(sampleset)
    results = get_results(sampleset, prob=lp)
    results = process_result(results)

    return results["feasible"], results


def compute_energy(sol: list, lp: LinearProg):
    data = sorted(list(lp.bqm.variables))
    sol_dict = {data[i]: sol[i] for i in range(len(sol))}
    s = 0
    for edge, value in lp.qubo[0].items():
        s += sol_dict[edge[0]] * sol_dict[edge[1]] * value
    return s


def save_ising_as_csv(quadratic_model, name, save_path, p: Optional[int] = None):
    p = 5 if p is None else p
    quadratic_model.to_bqm_qubo_ising(p)

    ising_linear = quadratic_model.ising[0]
    ising_quadratic = quadratic_model.ising[1]
    offset = quadratic_model.ising[2]
    variables = set()

    for k1, k2 in ising_quadratic.keys():
        variables.add(k1)
        variables.add(k2)
    renumerate = {var: num + 1 for num, var in enumerate(variables)}
    num_to_var = {v: k for k, v in renumerate.items()}

    spinglass_ising_linear = {}
    for key, value in ising_linear.items():
        spinglass_ising_linear[renumerate[key]] = value
    spinglass_ising_linear = dict(sorted(spinglass_ising_linear.items()))

    spinglass_ising_quadratic = {}
    for (key1, key2), value in ising_quadratic.items():
        spinglass_ising_quadratic[(renumerate[key1], renumerate[key2])] = value
    spinglass_ising_quadratic = dict(sorted(spinglass_ising_quadratic.items()))

    with open(os.path.join(save_path, f"{name}_ising.csv"), "w") as f:
        f.write(f"# offset: {offset}\n")
        for i, v in spinglass_ising_linear.items():
            f.write(f"{i} {i} {v}\n")
        for (i, j), v in spinglass_ising_quadratic.items():
            f.write(f"{i} {j} {v}\n")

    with open(os.path.join(save_path, f"{name}_ising_renumeration.pkl"), "wb") as f:
        data = [renumerate, num_to_var]
        pickle.dump(data, f)

