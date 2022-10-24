from collections import OrderedDict
import itertools
import networkx as nx
import pandas as pd
import numpy as np
from dimod import SampleSet


def create_stations_list(tracks: list[tuple]) -> list[str]:
    stations = []
    for track in tracks:
        for station in track:
            stations.append(station)

    return list(set(stations))


def agv_routes_as_edges(agv_routes: dict[int, tuple]) -> dict[int, list]:  # TODO write test
    return_dict = {}
    for j in agv_routes.keys():
        if len(agv_routes[j]) > 1:
            s_sp = [(agv_routes[j][i], agv_routes[j][i + 1]) for i in range(len(agv_routes[j]) - 1)]
            return_dict[j] = s_sp
    return return_dict


def create_agv_list(agv_routes: dict[int, tuple]) -> list[int]:
    return list(agv_routes.keys())


def create_same_way_dict(agv_routes: dict[int, tuple]) -> dict[tuple, list]:
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


def create_graph(tracks: list[tuple], agv_routes: dict[int, tuple]) -> nx.Graph:
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


def create_t_iterator(agv_routes: dict[int, tuple], in_out: str) -> list:
    t_iter = []  # list(itertools.product(J, stations))

    for j, route in agv_routes.items():
        for station in route:
            t_iter.append((in_out, j, station))
    return t_iter


def create_y_iterator(graph: nx.Graph) -> list[tuple]:
    y_iter = []
    agv_pass_through = nx.get_node_attributes(graph, "pass_through")
    for station in sorted(graph.nodes):
        if len(agv_pass_through[station]) >= 2:
            for pair in list(itertools.permutations(agv_pass_through[station], r=2)):
                y_iter.append((pair[0], pair[1], station))
    return y_iter


def create_z_iterator(graph: nx.Graph, agv_routes: dict[int, tuple]) -> list[tuple]:
    z_iter = []
    J = create_agv_list(agv_routes)
    agv_routes_as_edges_dict = agv_routes_as_edges(agv_routes)

    for j1, j2 in list(itertools.permutations(J, r=2)):
        if j1 in agv_routes_as_edges_dict.keys() and j2 in agv_routes_as_edges_dict.keys():
            for s, sp in agv_routes_as_edges_dict[j1]:
                if graph.number_of_edges(s, sp) < 2 and (sp, s) in agv_routes_as_edges_dict[j2]:
                    z_iter.append((j1, j2, s, sp))

    return z_iter


def create_iterators(graph: nx.Graph, agv_routes: dict[int, tuple]) -> dict[str, list]:

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
    for i in range(len(x_iter)):
        if vect[i] != 0:
            return_dict[x_iter[i]] = vect[i]

    return return_dict


def get_data4plot(res, agv_routes: dict, iterators: dict, complete_path, complete_path_rev, rev: list = []):

    x_iter = iterators["x"]
    sol = see_variables(res.x, x_iter)
    J = create_agv_list(agv_routes)
    times = {j: [] for j in J} 
    paths = {j: [] for j in J}
    
    for j in J:
        for s in agv_routes[j]:
            if j in rev:
                path = complete_path_rev
            else:
                path = complete_path
            for way in ["in", "out"]:   
                times[j].append(sol[(f'{way}', j, f'{s}')])
                paths[j].append(path[f"{s}_{way}"])
    return times, paths


def nice_print(res, agv_routes: dict, weights: dict, d_max: dict, v_in: dict, v_out: dict, iterators: dict):
    y_iter = iterators["y"]
    z_iter = iterators["z"]
    x_iter = iterators["x"]

    sol = see_variables(res.x, x_iter)
    J = create_agv_list(agv_routes)
    s_final = {j: agv_routes[j][-1] for j in J}
    L = {}
    contributions_sum = 0
    for j in J:
        for x in sol.keys():
            if x[0] == "in" and x[1] == j:
                L[x] = sol[x]
                for x2 in sol.keys():
                    if x2[0] == "out" and x2[1] == j and x2[2] == x[2]:
                        L[x2] = sol[x2]
                        continue
        v_out_j = v_out[j, s_final[j]]
        diff = L[('out', j, s_final[j])] - v_out[j, s_final[j]]
        contribution = weights[j] * (diff/d_max[j])
        contributions_sum += contribution
        print(f"{j} :", L, f"; v_out({j}, {s_final[j]}): {v_out_j} ; "
                           f"difference: {diff} ; contribution: {contribution}")
        L.clear()

    for y in y_iter:
        for x in sol.keys():
            if y == x:
                L[y] = sol[x]
                continue
    print("y :", L)
    L.clear()

    for z in z_iter:
        for x in sol.keys():
            if z == x:
                L[z] = sol[x]
                continue
    print("z :", L)
    L.clear()

    print("weights :", weights)
    print("d_max :", d_max)
    print("objective function :", res.fun)
    print("sum of contributions:", contributions_sum)


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

# def print_equations(A_ub: np.ndarray, b_ub: np.ndarray, A_eq: np.ndarray, b_eq, iterators):


def print_time(results: SampleSet):

    print(f"qpu_access_time: {results.info['qpu_access_time'] * 1e-6} seconds, "
          f"charge_time: {results.info['charge_time'] * 1e-6} seconds,"
          f" run_time: {results.info['run_time'] * 1e-6} seconds")


def print_best_feasible(results: SampleSet):
    feasible = results.filter(lambda d: d.is_feasible)
    print(feasible.first)


# DEPRECATED
def create_connectivity(agv: list, agv_dict: dict, s_sp: list) -> pd.DataFrame:
    connections_data = []
    for j in agv:
        temp = [1 if way_exist(agv_dict, j, way) else 0 for way in s_sp]
        connections_data.append(temp)
    connections = pd.DataFrame(connections_data, index=agv, columns=s_sp)
    return connections


def way_exist(agv_data: dict, agv: int, way: tuple) -> bool:
    track = agv_data[agv]
    for i in range(len(track)-1):
        if (track[i], track[i+1]) == way:
            return True
    return False
