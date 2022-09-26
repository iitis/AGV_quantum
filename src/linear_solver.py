from scipy.optimize import linprog
import itertools
import numpy as np
from src import utils
from typing import Optional


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


def create_precedence_matrix_z(iterators: dict):
    y_iter = iterators["y"]
    t_iter = iterators["t"]
    z_iter = iterators["z"]

    PVZ = []
    PVZ_b = []
    if len(z_iter)>0:
        for z1, z2 in itertools.combinations(z_iter, r=2):
            if z1[0] == z2[1] and z1[1] == z2[0] and z1[2] == z2[3] and z1[3] == z2[2]:
                t_vect = [0 for _ in t_iter]
                y_vect = [0 for _ in y_iter]
                z_vect = [1 if z == z1 or z == z2 else 0 for z in z_iter]
                PVZ.append(t_vect + y_vect + z_vect)
                PVZ_b.append(1)

    PVZ = np.array(PVZ)
    PVZ_b = np.array(PVZ_b)

    return PVZ, PVZ_b


def create_no_overtake_matrix(agv_routes: dict, tau_headway: dict, iterators: dict):
    t_iter = iterators["t"]
    y_iter = iterators["y"]
    z_iter = iterators["z"]

    agv_routes_as_edges = utils.agv_routes_as_edges(agv_routes)

    NO = []
    NO_b = []

    for j, jp, s, sp in z_iter:
        for z_s in [s, sp]:
            t_vect = [0 for _ in t_iter]
            y_vect = [-1 if y_j == j and y_jp == jp and y_s == z_s else 0 for y_j, y_jp, y_s in y_iter]
            z_vect = [1 if (j, jp, s, sp) == zp else 0 for zp in z_iter]
            NO.append(t_vect + y_vect + z_vect)
            NO_b.append(0)

    for j, jp, s, sp in tau_headway:
        t_vect = [0 for _ in t_iter]
        y_vect = [1 if y_j == j and y_jp == jp and y_s == s else -1 if y_j == j and y_jp == jp and y_s == sp else 0
                  for y_j, y_jp, y_s in y_iter]
        z_vect = [0 for _ in z_iter]
        NO.append(t_vect + y_vect + z_vect)
        NO_b.append(0)

    NO = np.array(NO)
    NO_b = np.array(NO_b)

    return NO, NO_b

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
    # TODO add condition s*
    for j1, j2, s, sp in tau_headway.keys():
        t_in_vect = [0 for _ in t_in_iter]
        t_out_vect = [1 if t == ("out", j1, s) else -1 if t == ("out", j2, s) else 0 for t in
                      t_out_iter]
        y_vect = [-1 * M if yp == (j2, j1, s) else 0 for yp in y_iter]
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
        for station in agv_routes[j]:
            t_in_vect = [1 if t == ("in", j, station) else 0 for t in t_in_iter]
            t_out_vect = [-1 if t == ("out", j, station) else 0 for t in t_out_iter]
            y_vect = [0 for _ in y_iter]
            z_vect = [0 for _ in z_iter]
            JC.append(t_in_vect + t_out_vect + y_vect + z_vect)
            JC_b.append(-1 * tau_operation[(j, station)])

    JC = np.array(JC)
    JC_b = np.array(JC_b)

    return JC, JC_b


def create_single_line_matrix(M, iterators):
    t_in_iter = iterators["t_in"]
    t_out_iter = iterators["t_out"]
    y_iter = iterators["y"]
    z_iter = iterators["z"]

    SL = []
    SL_b = []

    for j1, j2, s, sp in z_iter:
        t_in_vect = [1 if j_t == j1 and s_t == sp else 0 for _, j_t, s_t in t_in_iter]
        t_out_vect = [-1 if j_t == j2 and s_t == sp else 0 for _, j_t, s_t in t_out_iter]
        y_vect = [0 for _ in y_iter]
        z_vect = [-1 * M if zp == (j2, j1, sp, s) else 0 for zp in z_iter]
        SL.append(t_in_vect + t_out_vect + y_vect + z_vect)
        SL_b.append(0)

    SL = np.array(SL)
    SL_b = np.array(SL_b)

    return SL, SL_b


def create_bounds(v_in, v_out, d_max, iterators):
    # given = {("in", 0, "s0"): 0, ("in", 1, "s0"): 1}
    t_in_iter = iterators["t_in"]
    t_out_iter = iterators["t_out"]
    y_iter = iterators["y"]
    z_iter = iterators["z"]
    x_iter = iterators["x"]

    t_in_min = [v_in[(j, s)] for _, j, s in t_in_iter]
    t_in_max = [v_in[(j, s)] + d_max[j] for _, j, s in t_in_iter]

    t_out_min = [v_out[(j, s)] for _, j, s in t_out_iter]
    t_out_max = [v_out[(j, s)] + d_max[j] for _, j, s in t_out_iter]

    y_min = [0 for _ in y_iter]
    y_max = [1 for _ in y_iter]

    z_min = [0 for _ in z_iter]
    z_max = [1 for _ in z_iter]

    x_min = t_in_min + t_out_min + y_min + z_min
    x_max = t_in_max + t_out_max + y_max + z_max

    bounds = [(x_min[i], x_max[i]) for i in range(len(x_iter))]
    return bounds


def solve(M: int, tracks: list, tracks_len: dict, agv_routes: dict, d_max: dict,
          tau_pass: dict, tau_headway: dict, tau_operation: dict, weights: dict, initial_conditions: Optional[dict]):

    stations = utils.create_stations_list(tracks)
    J = utils.create_agv_list(agv_routes)
    graph = utils.create_graph(tracks, agv_routes)

    iterators = utils.create_iterators(graph, agv_routes)

    initial_conditions = {t_in: 0 for t_in in iterators["t_in"]} if not initial_conditions else initial_conditions
    v_in, v_out = utils.create_v_in_out(tracks_len, agv_routes, tau_operation, iterators, initial_conditions)

    PVY, PVY_b = create_precedence_matrix_y(iterators)
    PVZ, PVZ_b = create_precedence_matrix_z(iterators)
    NO, NO_b = create_no_overtake_matrix(agv_routes, tau_headway, iterators)

    MPT, MPT_b = create_minimal_passing_time_matrix(agv_routes, tau_pass, iterators)
    MH, MH_b = create_minimal_headway_matrix(M, tracks, agv_routes, tau_headway, iterators)
    JC, JC_b = create_junction_condition_matrix(M, tracks, agv_routes, tau_operation, iterators)
    SL, SL_b = create_single_line_matrix(M, iterators)

    bounds = create_bounds(v_in, v_out, d_max, iterators)

    if MPT.size >= 2 and MH.size >= 2:  # TO DO more sensible, for now is hack
        if SL.size>0:
            A_ub = np.concatenate((MPT, MH, JC, SL))
            b_ub = np.concatenate((MPT_b, MH_b, JC_b, SL_b))
        else:
            A_ub = np.concatenate((MPT, MH, JC))
            b_ub = np.concatenate((MPT_b, MH_b, JC_b))
    else:
        A_ub = JC
        b_ub = JC_b

    if NO.size > 0:
        if PVZ.size > 0:
            A_eq = np.concatenate((PVY, PVZ, NO))
            b_eq = np.concatenate((PVY_b, PVZ_b, NO_b))
        else:
            A_eq = np.concatenate((PVY, NO))
            b_eq = np.concatenate((PVY_b, NO_b))
    else:
        A_eq = PVY
        b_eq = PVY_b

    t_in = {}
    s_final = {j: agv_routes[j][-1] for j in J}
    obj = {("out", j, s_final[j]): weights[j]/d_max[j] for j in J}
    c = [obj[v] if v in obj.keys() else 0 for v in iterators["x"]]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds = bounds, integrality=[1 for _ in iterators["x"]])
    return res, iterators

# it is moved by 2 units. I don't know why

