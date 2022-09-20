from scipy.optimize import linprog
import itertools
import numpy as np
from src import utils
import networkx as nx


def create_precedence_matrix_y(iterators: dict):
    y_iter = iterators["y"]
    t_iter = iterators["t"]
    z_iter = iterators["z"]
    PVY = []
    PVY_b = []

    for y1, y2 in itertools.combinations(y_iter, r=2):
        if y1[2] == y2[2] and y1[0] == y2[1] and y1[1] == y2[0]:
            t_vect = [0 for _ in t_iter]
            y_vect = [1 if (y == y1 or y == y2) else 0 for y in y_iter]
            z_vect = [0 for _ in z_iter]
            PVY.append(t_vect + y_vect + z_vect)
            PVY_b.append(1)

    PVY = np.array(PVY)
    PVY_b = np.array(PVY_b)

    return PVY, PVY_b


def create_minimal_passing_time_matrix(agv_routes, tau_pass, iterators):
    t_in_iter = iterators["t_in"]
    t_out_iter = iterators["t_out"]
    y_iter = iterators["y"]
    z_iter = iterators["z"]

    MPT = []
    MPT_b = []

    J = utils.create_agv_list(agv_routes)
    for j in J:
        s_sp = [(agv_routes[j][i], agv_routes[j][i + 1]) for i in range(len(agv_routes[j]) - 1)]
        for s, sp in s_sp:
            t_in_vect = [-1 if (t[1] == j and t[2] == sp) else 0 for t in t_in_iter]
            t_out_vect = [1 if (t[1] == j and t[2] == s) else 0 for t in t_out_iter]
            y_vect = [0 for _ in y_iter]
            z_vect = [0 for _ in z_iter]
            MPT.append(t_in_vect + t_out_vect + y_vect + z_vect)
            MPT_b.append(-1 * tau_pass[(j, s, sp)])

    MPT = np.array(MPT)
    MPT_b = np.array(MPT_b)

    return MPT, MPT_b


def create_minimal_headway_matrix(M: int, tracks: list[tuple], agv_routes: dict, tau_headway: dict, iterators: dict):

    if not all([len(track) > 1 for track in tracks]):
        return np.empty((0, 0)), np.empty((0, 0))

    t_in_iter = iterators["t_in"]
    t_out_iter = iterators["t_out"]
    y_iter = iterators["y"]
    z_iter = iterators["z"]

    MH = []
    MH_b = []

    for j1, j2, sy in y_iter:  # TO DO better and questions
        s_sp1 = [(agv_routes[j1][i], agv_routes[j1][i + 1]) for i in range(len(agv_routes[j1]) - 1)]
        s_sp2 = [(agv_routes[j2][i], agv_routes[j2][i + 1]) for i in range(len(agv_routes[j2]) - 1)]
        for s, sp in tracks:
            if (s, sp) in s_sp1 and (s, sp) in s_sp2:
                t_in_vect = [0 for _ in t_in_iter]
                t_out_vect = [1 if t == ("out", j1, sy) else -1 if t == ("out", j2, sy) else 0 for t in
                              t_out_iter]
                y_vect = [-1 * M if yp == (j2, j1, sy) else 0 for yp in y_iter]
                z_vect = [0 for _ in z_iter]

                MH.append(t_in_vect + t_out_vect + y_vect + z_vect)
                MH_b.append(-1 * tau_headway[(j1, j2, s, sp)])

    MH = np.array(MH)
    MH_b = np.array(MH_b)
    return MH, MH_b


def create_junction_condition_matrix(M, tracks, agv_routes, tau_operation, iterators):
    t_in_iter = iterators["t_in"]
    t_out_iter = iterators["t_out"]
    y_iter = iterators["y"]
    z_iter = iterators["z"]

    JC = []
    JC_b = []

    stations = utils.create_stations_list(tracks)
    J = utils.create_agv_list(agv_routes)

    for y in y_iter:
        t_in_vect = [-1 if t == ("in", y[1], y[2]) else 0 for t in t_in_iter]
        t_out_vect = [1 if t == ("out", y[0], y[2]) else 0 for t in t_out_iter]
        y_vect = [-1 * M if yp == (y[1], y[0], y[2]) else 0 for yp in y_iter]
        z_vect = [0 for _ in z_iter]
        JC.append(t_in_vect + t_out_vect + y_vect + z_vect)
        JC_b.append(0)

    for j in J:
        for station in stations:
            t_in_vect = [1 if t == ("in", j, station) else 0 for t in t_in_iter]
            t_out_vect = [-1 if t == ("out", j, station) else 0 for t in t_out_iter]
            y_vect = [0 for _ in y_iter]
            z_vect = [0 for _ in z_iter]
            JC.append(t_in_vect + t_out_vect + y_vect + z_vect)
            JC_b.append(-1 * tau_operation[(j, station)])

    JC = np.array(JC)
    JC_b = np.array(JC_b)

    return JC, JC_b


def create_bounds(initial_conditions, iterators):
    # given = {("in", 0, "s0"): 0, ("in", 1, "s0"): 1}
    given = initial_conditions
    t_in_iter = iterators["t_in"]
    t_out_iter = iterators["t_out"]
    y_iter = iterators["y"]
    z_iter = iterators["z"]
    x_iter = iterators["x"]

    t_in_min = [given[t] if t in given.keys() else 0 for t in t_in_iter]
    t_in_max = [None for _ in t_in_iter]

    t_out_min = [given[t] if t in given.keys() else 0 for t in t_out_iter]
    t_out_max = [given[t] if t in given.keys() else None for t in t_out_iter]

    y_min = [given[y] if y in given.keys() else 0 for y in y_iter]
    y_max = [1 for _ in y_iter]

    z_min = [0 for _ in z_iter]
    z_max = [1 for _ in z_iter]

    x_min = t_in_min + t_out_min + y_min + z_min
    x_max = t_in_max + t_out_max + y_max + z_max

    bounds = [(x_min[i], x_max[i]) for i in range(len(x_iter))]
    return bounds


def solve(M: int, tracks: list, agv_routes: dict, d_max: dict,
          tau_pass: dict, tau_headway: dict, tau_operation: dict, weights: dict):

    stations = utils.create_stations_list(tracks)
    J = utils.create_agv_list(agv_routes)
    graph = utils.create_graph(tracks, agv_routes)

    iterators = utils.create_iterators(graph, agv_routes)

    PVY, PVY_b = create_precedence_matrix_y(iterators)

    MPT, MPT_b = create_minimal_passing_time_matrix(agv_routes, tau_pass, iterators)
    MH, MH_b = create_minimal_headway_matrix(M, tracks, agv_routes, tau_headway, iterators)
    JC, JC_b = create_junction_condition_matrix(M, tracks, agv_routes, tau_operation, iterators)

    #bounds = create_bounds(initial_conditions, iterators)

    if MPT.size >= 2 and MH.size >= 2:  # TO DO more sensible, for now is hack

        A_ub = np.concatenate((MPT, MH, JC))
        b_ub = np.concatenate((MPT_b, MH_b, JC_b))
    else:
        A_ub = JC
        b_ub = JC_b

    A_eq = PVY
    b_eq = PVY_b

    t_in = {}
    s_final = "s0" if len(tracks[0]) == 1 else "s1"
    obj = {("out", 0, s_final): weights[0]/d_max[0], ("out", 1, s_final): weights[1] / d_max[1]}
    c = [obj[v] if v in obj.keys() else 0 for v in iterators["x"]]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, integrality=[1 for _ in iterators["x"]])
    return res, iterators["x"]

# it is moved by 2 units. I don't know why

