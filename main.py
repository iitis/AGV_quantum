from scipy.optimize import linprog
from math import inf
import itertools
import pandas as pd
import numpy as np
import utils
import networkx as nx

# INPUT PARAMETERS
M = 1000
num_of_stations = 2
tracks = [("s0", "s1")]
J_dict = {0: ("s0", "s1"), 1: ("s0", "s1"), 2: ("s1", "s0")}
d_max = {}
tau_pass = {}
tau_headway = {}
tau_operation = {}

# BASIC ITERATORS
stations = [f"s{i}" for i in range(num_of_stations)]
J = list(J_dict.keys())
j_jp = list(itertools.permutations(J, r=2))
s_sp = list(itertools.permutations(stations, r=2))
connectivity = utils.create_connectivity(J, J_dict, s_sp)

t_in_iter = utils.create_t_iterator(J, s_sp, connectivity)
t_out_iter = utils.create_t_iterator(J, s_sp, connectivity)
t_iter = t_in_iter + t_out_iter  # list(itertools.chain(t_in_iter, t_out_iter))

y_iter = utils.create_y_iterator(s_sp, connectivity)
z_iter = utils.create_z_iterator()

x_iter = t_iter + y_iter + z_iter

# for testing purposes
d_max = d_max if d_max else {i: 10 for i in J}
tau_pass = tau_pass if tau_pass else {(j, way[0], way[1]): 1 for j in J for way in s_sp}
tau_headway = tau_headway if tau_headway else {(order[0], order[1], way[0], way[1]): 1 for order in j_jp for way in s_sp}
tau_operation = tau_operation if tau_operation else {(j, station): 1 for j in J for station in stations}

# CREATE GRAPH
graph = nx.MultiGraph()
graph.add_nodes_from(stations)
graph.add_edges_from(tracks)

# EQUATIONS
# basic constrains

v_in = pd.DataFrame([[0, 0],
                     [0, 0],
                     [0, 0]], index=J, columns=stations)

v_out = pd.DataFrame([[0, 0],
                      [0, 0],
                      [0, 0]], index=J, columns=stations)

t_min = [v_in.at[t[0], t[1]] for t in t_in_iter] + [v_out.at[t[0], t[1]] for t in t_out_iter]
t_max = [v_in.at[t[0], t[1]] + d_max[t[0]] for t in t_in_iter] + \
        [v_out.at[t[0], t[1]] + d_max[t[0]] for t in t_out_iter]

y_min = [0 for _ in y_iter]
z_min = [0 for _ in z_iter]
y_max = [1 for _ in y_iter]
z_max = [1 for _ in z_iter]

x_min = t_min + y_min + z_min
x_max = t_max + y_max + z_max

bounds = [(x_min[i], x_max[i]) for i in range(len(x_iter))]

# precedence variables
PVY = []
PVZ = []

PVY = np.array(PVY)
PVZ = np.array(PVZ)

# minimal passing time
"""
MPT = []
MPT_b = []
for j in J:
    for way in s_sp:
        if connectivity.at[j, way] > 0:
            temp_in = [-1 if (t[0] == j and t[1] == way[1]) else 0 for t in t_in_iter]
            temp_out = [1 if (t[0] == j and t[1] == way[0]) else 0 for t in t_out_iter]
            temp = temp_in + temp_out
            MPT.append(temp)
            MPT_b.append(-1 * tau_pass.at[j, way])
del temp_in, temp_out, temp

MPT = np.array(MPT)
MPT_b = np.array(MPT_b)

# minimal headway
# eq (9) will be implemented later

MH = []
MH_b = []

for way in s_sp:
    temp_df = connectivity.loc[connectivity[way] > 0]
    if len(temp_df.index) <= 1: # there is no agvs that share way
        continue
    J_same_dir = list(itertools.permutations(list(temp_df[way].index), r=2))
    for pair in J_same_dir:
        temp = [1 if t == (pair[0], way[0]) else -1 if t == (pair[1], way[0]) else 0
                for t in t_out_iter]
        temp_in = [0 for _ in t_in_iter]
        MH.append(temp_in + temp)
        MH_b.append(M * y.at[(pair[1], pair[0]), way[0]] - tau_headway.at[pair, way])


del temp_df, J_same_dir

MH = np.array(MH)
MH_b = np.array(MH_b)

# single track line

STL = []
STL_b = []


# CREATE PARAMETERS FOR LINEAR OPTIMISATION

A_ub = np.concatenate((MPT, MH))
b_ub = np.concatenate((MPT_b, MH_b))

# OPTIMIZE

res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, integrality=[1 for _ in x_iter])

"""


