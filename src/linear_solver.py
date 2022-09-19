from scipy.optimize import linprog
import itertools
import numpy as np
from src import utils
import networkx as nx


def create_precedence_matrix_y(iterators: dict):
    y_iter = iterators["y"]
    t_iter = iterators["t"]
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
    return PVY, PVY_b


def create_minimal_passing_time_matrix(J, s_sp, connectivity, tau_pass, iterators):
    t_in_iter = iterators["t_in"]
    t_out_iter = iterators["t_out"]
    y_iter = iterators["y"]

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

    MPT = np.array(MPT)
    MPT_b = np.array(MPT_b)
    return MPT, MPT_b


def create_bounds(initial_conditions, iterators):
    # given = {("in", 0, "s0"): 0, ("in", 1, "s0"): 1}
    given = initial_conditions
    t_in_iter = iterators["t_in"]
    t_out_iter = iterators["t_out"]
    y_iter = iterators["y"]
    x_iter = iterators["x"]

    y_min = [given[y] if y in given.keys() else 0 for y in y_iter]
    y_max = [given[y] if y in given.keys() else 1 for y in y_iter]

    t_in_min = [given[t] if t in given.keys() else 0 for t in t_in_iter]
    t_in_max = [given[t] if t in given.keys() else None for t in t_in_iter]

    t_out_min = [given[t] if t in given.keys() else 0 for t in t_out_iter]
    t_out_max = [given[t] if t in given.keys() else None for t in t_out_iter]

    x_min = t_in_min + t_out_min + y_min
    x_max = t_in_max + t_out_max + y_max

    bounds = [(x_min[i], x_max[i]) for i in range(len(x_iter))]
    return bounds


def create_minimal_headway_matrix(M, s_sp, connectivity, tau_headway, iterators):

    t_in_iter = iterators["t_in"]
    t_out_iter = iterators["t_out"]
    y_iter = iterators["y"]

    MH = []
    MH_b = []

    for way in s_sp:
        temp_df = connectivity.loc[connectivity[way] > 0]
        if len(temp_df.index) <= 1:  # there is no agvs that share this way
            continue
        J_same_dir = list(itertools.permutations(list(temp_df[way].index), r=2))
        for pair in J_same_dir:
            t_out_vect = [1 if t == ("out", pair[0], way[0]) else -1 if t == ("out", pair[1], way[0]) else 0 for t in
                          t_out_iter]
            t_in_vect = [0 for _ in t_in_iter]
            y_vect = [-1 * M if y == (pair[1], pair[0], way[0]) else 0 for y in y_iter]
            MH.append(t_in_vect + t_out_vect + y_vect)
            MH_b.append(-1 * tau_headway[(pair[0], pair[1], way[0], way[1])])

    del temp_df, J_same_dir, t_out_vect, t_in_vect, y_vect

    MH = np.array(MH)
    MH_b = np.array(MH_b)
    return MH, MH_b


# junction condition

def create_junction_condition_matrix(M, J, j_jp, stations, tau_operation, iterators):
    t_in_iter = iterators["t_in"]
    t_out_iter = iterators["t_out"]
    y_iter = iterators["y"]

    JC = []
    JC_b = []

    for pair in j_jp:
        for station in stations:
            t_in_vect = [-1 if t == ("in", pair[1], station) else 0 for t in t_in_iter]
            t_out_vect = [1 if t == ("out", pair[0], station) else 0 for t in t_out_iter]
            y_vect = [-1 * M if y == (pair[1], pair[0], station) else 0 for y in y_iter]
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

    return JC, JC_b


def solve(M: int, tracks: list, agv_routes: dict, d_max: dict,
          tau_pass: dict, tau_headway: dict, tau_operation: dict, weights: dict, initial_conditions: dict):

    stations = utils.create_stations_list(tracks)
    J = utils.create_agv_list(agv_routes)
    graph = utils.create_graph(tracks, stations)

    # BASIC ITERATORS
    j_jp = list(itertools.permutations(J, r=2))
    s_sp = list(itertools.permutations(stations, r=2))
    connectivity = utils.create_connectivity(J, agv_routes, s_sp)

    iterators = utils.create_iterators(J, s_sp, connectivity)

    PVY, PVY_b = create_precedence_matrix_y(iterators)

    MPT, MPT_b = create_minimal_passing_time_matrix(J, s_sp, connectivity, tau_pass, iterators)
    MH, MH_b = create_minimal_headway_matrix(M, s_sp, connectivity, tau_headway, iterators)
    JC, JC_b = create_junction_condition_matrix(M, J, j_jp, stations, tau_operation, iterators)

    bounds = create_bounds(initial_conditions, iterators)


    A_ub = np.concatenate((MPT, MH, JC))
    b_ub = np.concatenate((MPT_b, MH_b, JC_b))

    A_eq = PVY
    b_eq = PVY_b

    t_in = {}
    obj = {("out", 0, "s1"): weights[0]/d_max[0], ("out", 1, "s0"): weights[1] / d_max[1]}
    c = [obj[v] if v in obj.keys() else 0 for v in iterators["x"]]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, integrality=[1 for _ in iterators["x"]])
    return res, iterators["x"]

# it is moved by 2 units. I don't know why

