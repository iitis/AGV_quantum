from scipy.optimize import linprog
from math import inf
import itertools
import pandas as pd
import numpy as np
import helpers

# PARAMETERS
M = 1000
num_of_stations = 2

J_dict = {0: ("s0", "s1"), 1: ("s0", "s1"), 2: ("s1", "s0")}
d_max = [10 for _ in J_dict.keys()]

# CREATE DATA
J = list(J_dict.keys())
j_jp = list(itertools.permutations(J, r=2))


stations = [f"s{i}" for i in range(num_of_stations)]
s_sp = list(itertools.permutations(stations, r=2))

y_data = []
for order in j_jp:
    temp = [1 if order[0] < order[1] else 0 for _ in stations]  # arbitrary order, lower number goes first
    y_data.append(temp)
del temp
y = pd.DataFrame(y_data, index=j_jp, columns=stations)

z = pd.DataFrame(index=j_jp, columns=s_sp)

connections_data = []
for j in J:
    temp = [1 if helpers.way_exist(J_dict, j, way) else 0 for way in s_sp]
    connections_data.append(temp)
del temp
connections = pd.DataFrame(connections_data, index=J, columns=s_sp)

# here tau_pass also encode existing paths as tau_pass(j, s, s') > 0 for all j, s, s'
# alternatively we may introduce another dataframe with the same columns and rows
tau_pass = pd.DataFrame([[1, -inf],
                         [1, -inf],
                         [-inf, 1]], index=J, columns=s_sp)

tau_headway = pd.DataFrame(np.ones([len(j_jp), len(s_sp)]), index=j_jp, columns=s_sp)
tau_operation = pd.DataFrame(np.ones([len(J), len(stations)]), index=J, columns=stations)

# EQUATIONS
# I don't know yet if all those iterators will be needed

t_in_iter = []  # list(itertools.product(J, stations))
for j, way in itertools.product(J, s_sp):
    if connections.at[j, way] > 0:
        t_in_iter.append((j, way[0]))
        t_in_iter.append((j, way[1]))
t_out_iter = t_in_iter  # list(itertools.product(J, stations))
t_iter = t_in_iter + t_out_iter  # list(itertools.chain(t_in_iter, t_out_iter))
print(t_iter)

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

t_min = np.array(t_min)
t_max = np.array(t_max)

# minimal passing time

MPT = []
MPT_b = []
for j in J:
    for way in s_sp:
        if connections.at[j, way] > 0:
            temp_in = [-1 if (t[0] == j and t[1] == way[1]) else 0 for t in t_in_iter]
            temp_out = [1 if (t[0] == j and t[1] == way[0]) else 0 for t in t_out_iter]
            temp = temp_in + temp_out
            MPT.append(temp)
            MPT_b.append(tau_pass.at[j, way])
del temp_in, temp_out, temp

MPT = np.array(MPT)
MPT_b = np.array(MPT_b)

# minimal headway
# eq (9) will be implemented later

MH = []
MH_b = []

for way in s_sp:
    temp_df = connections.loc[connections[way] > 0]



# CREATE PARAMETERS FOR LINEAR OPTIMISATION

A_ub = MPT
b_ub = MPT_b


