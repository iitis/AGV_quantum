from math import inf
import itertools
import pandas as pd
import numpy as np


def create_y(j_jp, stations):
    y_data = []
    for order in j_jp:
        temp = [1 if order[0] < order[1] else 0 for _ in stations]  # arbitrary order, lower number goes first
        y_data.append(temp)
    del temp
    y = pd.DataFrame(y_data, index=j_jp, columns=stations)
    return y


def way_exist(agv_data: dict, agv: int, way: tuple) -> bool:
    track = agv_data[agv]
    for i in range(len(track)-1):
        if (track[i], track[i+1]) == way:
            return True

    return False


def find_agvs_heading_opposite(connections: pd.DataFrame, single_track_list: list) -> list:
    for way in list(connections.columns):
        temp_df = connections.loc[connections[way] > 0]

