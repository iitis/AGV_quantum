from math import inf
import itertools
import pandas as pd
import numpy as np


def create_connectivity(agv: list, agv_dict: dict, s_sp: list) -> pd.DataFrame:
    connections_data = []
    for j in agv:
        temp = [1 if way_exist(agv_dict, j, way) else 0 for way in s_sp]
        connections_data.append(temp)
    connections = pd.DataFrame(connections_data, index=agv, columns=s_sp)
    return connections


def create_t_iterator(agv:list, s_sp: list, connectivity: pd.DataFrame) -> list:
    t_iter = []  # list(itertools.product(J, stations))
    for j, way in itertools.product(agv, s_sp):
        if connectivity.at[j, way] > 0:
            t_iter.append((j, way[0]))
            t_iter.append((j, way[1]))
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
    return y_iter


def create_z_iterator(s_sp: list, connectivity: pd.DataFrame) -> list:
    z_iter = []
    for way in s_sp:
        pass
    return []
