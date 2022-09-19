from math import inf
import itertools
import networkx as nx
import pandas as pd
import numpy as np


def create_stations_list(tracks: list[tuple]) -> list:
    stations = []
    for track in tracks:
        for station in track:
            stations.append(station)

    return list(set(stations))


def create_agv_list(agv_routes: dict) -> list:
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





def way_exist(agv_data: dict, agv: int, way: tuple) -> bool:
    track = agv_data[agv]
    for i in range(len(track)-1):
        if (track[i], track[i+1]) == way:
            return True
    return False


def delete_none(_dict: dict) -> dict:
    """Delete None values recursively from all of the dictionaries, tuples, lists, sets"""
    if isinstance(_dict, dict):
        for key, value in list(_dict.items()):
            if isinstance(value, (list, dict, tuple, set)):
                _dict[key] = delete_none(value)
            elif value is None or key is None:
                del _dict[key]

    elif isinstance(_dict, (list, set, tuple)):
        _dict = type(_dict)(delete_none(item) for item in _dict if item is not None)

    return _dict


def create_y_iterator(s_sp: list, connectivity: pd.DataFrame) -> list:
    y_iter = []
    for way in s_sp:
        temp_df = connectivity.loc[connectivity[way] > 0]
        if len(temp_df.index) <= 1:  # there is no agvs that share way
            continue
        J_same_dir = list(itertools.permutations(list(temp_df[way].index), r=2))
        for pair in J_same_dir:
            y_iter.append((pair[0], pair[1], way[0]))
            y_iter.append((pair[0], pair[1], way[1]))
    return y_iter


def create_z_iterator(graph: nx.Graph, s_sp: list, connectivity: pd.DataFrame) -> list:
    z_iter = []
    for way in s_sp:
        if graph.number_of_edges(way[0], way[1]) == 1:
            temp_df = connectivity.loc[connectivity[way] > 0]
            one = list(temp_df[way].index)
            temp_df_2 = connectivity.loc[connectivity[way[::-1]] > 0]
            print(temp_df_2)
            two = list(temp_df_2[way[::-1]])
            pairs = list(itertools.product(one, two))

    return one


def create_iterators(agv: list, s_sp: list, connectivity: pd.DataFrame):
    y_iter = create_y_iterator(s_sp, connectivity)

    t_in_iter = create_t_iterator(agv, s_sp, connectivity, "in")
    t_out_iter = create_t_iterator(agv, s_sp, connectivity, "out")
    t_iter = t_in_iter + t_out_iter

    x_iter = t_iter + y_iter

    return {"t_in": t_in_iter, "t_out": t_out_iter, "t": t_iter, "y": y_iter, "x": x_iter}


def see_variables(vect: list, x_iter: list) -> dict:
    return {x_iter[i]: vect[i] for i in range(len(x_iter))}



def create_connectivity(agv: list, agv_dict: dict, s_sp: list) -> pd.DataFrame:
    connections_data = []
    for j in agv:
        temp = [1 if way_exist(agv_dict, j, way) else 0 for way in s_sp]
        connections_data.append(temp)
    connections = pd.DataFrame(connections_data, index=agv, columns=s_sp)
    return connections
