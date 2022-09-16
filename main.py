from scipy.optimize import linprog
import itertools
import pandas as pd
import numpy as np
import utils
import networkx as nx

#  QUICK AND DIRTY EXAMPLE 1

# INPUT PARAMETERS
M = 50
num_of_stations = 2
tracks = [("s0", "s1"), ("s1", "s0")]
J_dict = {0: ("s0", "s1"), 1: ("s0", "s1")}

stations = list(set([station for track in tracks for station in track]))
J = list(J_dict.keys())

# CREATE GRAPH
graph = nx.MultiGraph()
graph.add_nodes_from(stations)
graph.add_edges_from(tracks)

d_max = {i: 5 for i in J}
tau_pass = {(agv, way[0], way[1]): 5 for agv, way in J_dict.items()}
tau_headway = {(0, 1, "s0", "s1"): 2, (1, 0, "s0", "s1"): 2}
tau_operation = {(agv, station): 1 for agv in J for station in stations}


# BASIC ITERATORS
j_jp = list(itertools.permutations(J, r=2))
s_sp = list(itertools.permutations(stations, r=2))
connectivity = utils.create_connectivity(J, J_dict, s_sp)

y_iter = utils.create_y_iterator(s_sp, connectivity)

t_in_iter = utils.create_t_iterator(J, s_sp, connectivity, "in")
t_out_iter = utils.create_t_iterator(J, s_sp, connectivity, "out")
t_iter = t_in_iter + t_out_iter

x_iter = t_iter + y_iter

# EQUATIONS
# basic constrains
#given = {("in", 0, "s0"): 0, ("in", 1, "s0"): 1}
given = {}


y_min = [given[y] if y in given.keys() else 0 for y in y_iter]
y_max = [given[y] if y in given.keys() else 1 for y in y_iter]

t_in_min = [given[t] if t in given.keys() else 0 for t in t_in_iter]
t_in_max = [given[t] if t in given.keys() else None for t in t_in_iter]

t_out_min = [given[t] if t in given.keys() else 0 for t in t_out_iter]
t_out_max = [given[t] if t in given.keys() else None for t in t_out_iter]

x_min = t_in_min + t_out_min + y_min
x_max = t_in_max + t_out_max + y_max

bounds = [(x_min[i], x_max[i]) for i in range(len(x_iter))]

# precedence variables
PVY = []
PVY_b = []


for pair in itertools.combinations(y_iter, r=2):
    if pair[0][2] == pair[1][2] and pair[0][0] == pair[1][1] and pair[0][1] == pair[1][0]:
        t_vect = [0 for _ in t_iter]
        y_vect = [1 if (y == pair[0] or y == pair[1]) else 0 for y in y_iter]
        PVY.append(t_vect + y_vect)
        PVY_b.append(1)
del t_vect, y_vect

PVY = np.array(PVY)
PVY_b = np.array(PVY_b)

# PVZ = []
# PVZ = np.array(PVZ)

# minimal passing time
MPT = []
MPT_b = []
for j in J:
    for way in s_sp:
        if connectivity.at[j, way] > 0:
            t_in_vect = [-1 if (t[1] == j and t[2] == way[1]) else 0 for t in t_in_iter]
            t_out_vect = [1 if (t[1] == j and t[2] == way[0]) else 0 for t in t_out_iter]
            y_vect = [0 for _ in y_iter]
            temp = t_in_vect + t_out_vect + y_vect
            MPT.append(temp)
            MPT_b.append(-1 * tau_pass[(j, way[0], way[1])])
del t_in_vect, t_out_vect, temp

MPT = np.array(MPT)
MPT_b = np.array(MPT_b)

# minimal headway
# eq (9) will be implemented later
MH = []
MH_b = []

for way in s_sp:
    temp_df = connectivity.loc[connectivity[way] > 0]
    if len(temp_df.index) <= 1:  # there is no agvs that share this way
        continue
    J_same_dir = list(itertools.permutations(list(temp_df[way].index), r=2))
    for pair in J_same_dir:
        t_out_vect = [1 if t == ("out", pair[0], way[0]) else -1 if t == ("out", pair[1], way[0]) else 0 for t in t_out_iter]
        t_in_vect = [0 for _ in t_in_iter]
        y_vect = [-1 * M if y == (pair[1], pair[0], way[0]) else 0 for y in y_iter]
        MH.append(t_in_vect + t_out_vect + y_vect)
        MH_b.append(-1 * tau_headway[(pair[0], pair[1], way[0], way[1])])

del temp_df, J_same_dir, t_out_vect, t_in_vect, y_vect

MH = np.array(MH)
MH_b = np.array(MH_b)

# junction condition
JC = []
JC_b = []

for pair in j_jp:
    for station in stations:
        t_in_vect = [-1 if t == ("in", pair[1], station) else 0 for t in t_in_iter]
        t_out_vect = [1 if t == ("out", pair[0], station) else 0 for t in t_out_iter]
        y_vect = [-1*M if y == (pair[1], pair[0], station) else 0 for y in y_iter]
        JC.append(t_in_vect + t_out_vect + y_vect)
        JC_b.append(0)

for j in J:
    for station in stations:
        t_in_vect = [1 if t == ("in", j, station) else 0 for t in t_in_iter]
        t_out_vect = [-1 if t == ("out", j, station) else 0 for t in t_out_iter]
        y_vect = [0 for _ in y_iter]
        JC.append(t_in_vect + t_out_vect + y_vect)
        JC_b.append(-1 ^ tau_operation[(j, station)])

JC = np.array(JC)
JC_b = np.array(JC_b)


"""
v_in = pd.DataFrame([[0, 0],
                     [0, 0],
                     [0, 0]], index=J, columns=stations)

v_out = pd.DataFrame([[0, 0],
                      [0, 0],
                      [0, 0]], index=J, columns=stations)

t_min = [v_in.at[t[0], t[1]] for t in t_in_iter] + [v_out.at[t[0], t[1]] for t in t_out_iter]
t_max = [v_in.at[t[0], t[1]] + d_max[t[0]] for t in t_in_iter] + \
        [v_out.at[t[0], t[1]] + d_max[t[0]] for t in t_out_iter]
"""


# CREATE PARAMETERS FOR LINEAR OPTIMISATION
A_ub = np.concatenate((MPT, MH, JC))
b_ub = np.concatenate((MPT_b, MH_b, JC_b))

A_eq = PVY
b_eq = PVY_b

t_in = {}
obj = {("out", 0, "s1"): 1/5, ("out", 1, "s0"): 1/5}
# OPTIMIZE

c = [obj[v] if v in obj.keys() else 0 for v in x_iter]
res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq = A_eq, b_eq=b_eq, bounds=bounds, integrality=[1 for _ in x_iter])
print(res.fun)
print(utils.see_variables(res.x, x_iter))

# it is moved by 2 units. I don't know why

