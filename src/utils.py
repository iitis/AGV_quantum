from math import inf
import itertools
import networkx as nx
import pandas as pd
import numpy as np


def create_stations_list(tracks: list[tuple]) -> list[str]:
    stations = []
    for track in tracks:
        for station in track:
            stations.append(station)

    return list(set(stations))


def create_agv_list(agv_routes: dict[int, tuple]) -> list[int]:
    return list(agv_routes.keys())


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
    for station in graph.nodes:
        if len(agv_pass_through[station]) >= 2:
            for pair in list(itertools.permutations(agv_pass_through[station], r=2)):
                y_iter.append((pair[0], pair[1], station))
    return y_iter


def create_z_iterator(graph: nx.Graph, agv_routes: dict[int, tuple]) -> list[tuple]:
    z_iter = []
    J = create_agv_list(agv_routes)
    agv_routes_as_edges = {}
    for j in J:
        if len(agv_routes[j]) > 1:
            s_sp = [(agv_routes[j][i], agv_routes[j][i+1]) for i in range(len(agv_routes[j])-1)]
            agv_routes_as_edges[j] = s_sp

    for j1, j2 in list(itertools.permutations(J, r=2)):
        if j1 in agv_routes_as_edges.keys() and j2 in agv_routes_as_edges.keys():
            for s, sp in agv_routes_as_edges[j1]:
                if graph.number_of_edges(s, sp) < 2 and (sp, s) in agv_routes_as_edges[j2]:
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
