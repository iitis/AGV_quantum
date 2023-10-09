"""implementation of linear solver on ILP"""

import itertools
from typing import Optional
import numpy as np
from docplex.mp.model import Model
from docplex.mp.solution import SolveSolution

from AGV_quantum import see_non_zero_variables, create_graph, create_iterators, create_agv_list



class LinearAGV:
    """
    Describes given problem as linear programing described in the paper
    """
    def __init__(self, M: int, tracks: list, tracks_len: dict, agv_routes: dict, d_max: dict,
                 tau_pass: dict, tau_headway: dict, tau_operation: dict, weights: dict,
                 initial_conditions: Optional[dict] = None):

        _graph = create_graph(tracks, agv_routes)
        _iterators = create_iterators(_graph, agv_routes)

        self.J = create_agv_list(agv_routes)

        self.t_in_iter = _iterators["t_in"]
        self.t_out_iter = _iterators["t_out"]
        self.t_iter = _iterators["t"]
        self.y_iter = _iterators["y"]
        self.z_iter = _iterators["z"]
        self.x_iter = _iterators["x"]

        self.initial_conditions = {t_in: 0 for t_in in self.t_in_iter} if not initial_conditions \
            else initial_conditions

        self.v_in, self.v_out = self._create_v_in_out(tracks_len, agv_routes, tau_operation, initial_conditions)

        PVY, PVY_b = self._create_precedence_matrix_y()
        PVZ, PVZ_b = self._create_precedence_matrix_z()
        NO, NO_b = self._create_no_overtake_matrix(tau_headway)

        MPT, MPT_b = self._create_minimal_passing_time_matrix(agv_routes, tau_pass)
        MH, MH_b = self._create_minimal_headway_matrix(M, tracks, tau_headway)
        JC, JC_b = self._create_junction_condition_matrix(M, agv_routes, tau_operation)
        SL, SL_b = self._create_single_line_matrix(M)

        self.bounds = self._create_bounds(d_max)

        if MPT.size >= 2 and MH.size >= 2:  # TO DO more sensible, for now is hack
            if SL.size > 0:
                self.A_ub = np.concatenate((MPT, MH, JC, SL))
                self.b_ub = np.concatenate((MPT_b, MH_b, JC_b, SL_b))
            else:
                self.A_ub = np.concatenate((MPT, MH, JC))
                self.b_ub = np.concatenate((MPT_b, MH_b, JC_b))
        elif MPT.size >= 2:
            if SL.size > 0:
                self.A_ub = np.concatenate((MPT, JC, SL))
                self.b_ub = np.concatenate((MPT_b, JC_b, SL_b))
            else:
                self.A_ub = np.concatenate((MPT, JC))
                self.b_ub = np.concatenate((MPT_b, JC_b))
        else:
            self.A_ub = JC
            self.b_ub = JC_b

        if NO.size > 0:
            if PVZ.size > 0:
                self.A_eq = np.concatenate((PVY, PVZ, NO))
                self.b_eq = np.concatenate((PVY_b, PVZ_b, NO_b))
            else:
                self.A_eq = np.concatenate((PVY, NO))
                self.b_eq = np.concatenate((PVY_b, NO_b))
        else:
            self.A_eq = PVY
            self.b_eq = PVY_b

        s_final = {j: agv_routes[j][-1] for j in self.J}
        obj = {("out", j, s_final[j]): weights[j] / d_max[j] for j in self.J}
        self.c = [obj[v] if v in obj.keys() else 0 for v in self.x_iter]

    def create_linear_model(self, num_threads: int = None) -> Model:

        model = Model(name='linear_programing_AGV')

        lower_bounds = [bound[0] for bound in self.bounds]
        upper_bounds = [bound[1] for bound in self.bounds]

        if num_threads:
            model.context.cplex_parameters.threads = num_threads

        variables = model.integer_var_dict(self.t_iter, lb=lower_bounds[0:len(self.t_iter)],
                                           ub=upper_bounds[0:len(self.t_iter)], name="t_", key_format="%s")
        if self.y_iter:
            y_variables = model.binary_var_dict(self.y_iter, name="y_", key_format="%s")
            variables = variables | y_variables

        if self.z_iter:
            z_variables = model.binary_var_dict(self.z_iter, name="z_", key_format="%s")
            variables = variables | z_variables

        for index, row in enumerate(self.A_ub):
            non_zero = see_non_zero_variables(row, self.x_iter)
            # print(f"{sum([variables[var_name] * sign for var_name, sign in non_zero.items()])} <= {b_ub[index]}")
            model.add_constraint(
                sum([variables[var_name] * sign for var_name, sign in non_zero.items()]) <= self.b_ub[index])

        for index, row in enumerate(self.A_eq):
            non_zero = see_non_zero_variables(row, self.x_iter)
            # print(f"{sum([variables[var_name] * sign for var_name, sign in non_zero.items()])} == {b_eq[index]}")
            model.add_constraint(
                sum([variables[var_name] * sign for var_name, sign in non_zero.items()]) == self.b_eq[index])

        obj_dict = {self.x_iter[i]: self.c[i] for i in range(len(self.c))}
        # print(sum([variables[var_name] * sign for var_name, sign in obj_dict.items()]))
        model.minimize(sum([variables[var_name] * sign for var_name, sign in obj_dict.items()]))
        return model

    def nice_print(self, model: Model, sol: SolveSolution):
        # WIP
        for var in model.iter_variables():
            print(var, sol.get_var_value(var))


    def _create_v_in_out(self, tracks_len: dict, agv_routes: dict, tau_operation: dict,
                        initial_conditions: dict):

        if not tracks_len:
            return {(j, s): initial_conditions[("in", j, s)] for _, j, s in self.t_in_iter}, \
                   {(j, s): initial_conditions[("in", j, s)] + tau_operation[j, s] for _, j, s in self.t_in_iter}

        v = {}

        for j in self.J:

            for i, s in enumerate(agv_routes[j]):
                if i == 0:
                    v[(j, s)] = initial_conditions[("in", j, s)]
                else:
                    s_before = agv_routes[j][i - 1]
                    v[(j, s)] = v[(j, agv_routes[j][i - 1])] + tau_operation[(j, s_before)] + tracks_len[(s_before, s)]

        v_in = v
        v_out = {(j, s): v_in[(j, s)] + tau_operation[(j, s)] for _, j, s in self.t_in_iter}
        return v_in, v_out

    def _create_precedence_matrix_y(self):

        PVY = []
        PVY_b = []

        for y1, y2 in itertools.combinations(self.y_iter, r=2):
            if y1[2] == y2[2] and y1[0] == y2[1] and y1[1] == y2[0]:
                t_vect = [0 for _ in self.t_iter]
                y_vect = [1 if y in (y1, y2) else 0 for y in self.y_iter]
                z_vect = [0 for _ in self.z_iter]
                PVY.append(t_vect + y_vect + z_vect)
                PVY_b.append(1)

        PVY = np.array(PVY)
        PVY_b = np.array(PVY_b)

        return PVY, PVY_b

    def _create_precedence_matrix_z(self):

        PVZ = []
        PVZ_b = []
        if len(self.z_iter) > 0:
            for z1, z2 in itertools.combinations(self.z_iter, r=2):
                if z1[0] == z2[1] and z1[1] == z2[0] and z1[2] == z2[3] and z1[3] == z2[2]:
                    t_vect = [0 for _ in self.t_iter]
                    y_vect = [0 for _ in self.y_iter]
                    z_vect = [1 if z in (z1, z2) else 0 for z in self.z_iter]
                    PVZ.append(t_vect + y_vect + z_vect)
                    PVZ_b.append(1)

        PVZ = np.array(PVZ)
        PVZ_b = np.array(PVZ_b)

        return PVZ, PVZ_b

    def _create_no_overtake_matrix(self, tau_headway: dict):

        NO = []
        NO_b = []

        for j, jp, s, sp in self.z_iter:
            for z_s in [s, sp]:
                t_vect = [0 for _ in self.t_iter]
                y_vect = [-1 if y_j == j and y_jp == jp and y_s == z_s else 0 for y_j, y_jp, y_s in self.y_iter]
                z_vect = [1 if (j, jp, s, sp) == zp else 0 for zp in self.z_iter]
                NO.append(t_vect + y_vect + z_vect)
                NO_b.append(0)

        for j, jp, s, sp in tau_headway:
            t_vect = [0 for _ in self.t_iter]
            y_vect = [1 if y_j == j and y_jp == jp and y_s == s else -1 if y_j == j and y_jp == jp and y_s == sp else 0
                      for y_j, y_jp, y_s in self.y_iter]
            z_vect = [0 for _ in self.z_iter]
            NO.append(t_vect + y_vect + z_vect)
            NO_b.append(0)

        NO = np.array(NO)
        NO_b = np.array(NO_b)

        return NO, NO_b

    def _create_minimal_passing_time_matrix(self, agv_routes, tau_pass):

        MPT = []
        MPT_b = []

        J = create_agv_list(agv_routes)
        for j in J:
            s_sp = [(agv_routes[j][i], agv_routes[j][i + 1]) for i in range(len(agv_routes[j]) - 1)]
            for s, sp in s_sp:
                t_in_vect = [-1 if (t[1] == j and t[2] == sp) else 0 for t in self.t_in_iter]
                t_out_vect = [1 if (t[1] == j and t[2] == s) else 0 for t in self.t_out_iter]
                y_vect = [0 for _ in self.y_iter]
                z_vect = [0 for _ in self.z_iter]
                MPT.append(t_in_vect + t_out_vect + y_vect + z_vect)
                MPT_b.append(-1 * tau_pass[(j, s, sp)])

        MPT = np.array(MPT)
        MPT_b = np.array(MPT_b)

        return MPT, MPT_b

    def _create_minimal_headway_matrix(self, M: int, tracks, tau_headway: dict):

        if not all([len(track) > 1 for track in tracks]):
            return np.empty((0, 0)), np.empty((0, 0))

        MH = []
        MH_b = []
        for j1, j2, s, sp in tau_headway.keys():
            t_in_vect = [0 for _ in self.t_in_iter]
            t_out_vect = [1 if t == ("out", j1, s) else -1 if t == ("out", j2, s) else 0 for t in
                          self.t_out_iter]
            y_vect = [-1 * M if yp == (j2, j1, s) else 0 for yp in self.y_iter]
            z_vect = [0 for _ in self.z_iter]

            MH.append(t_in_vect + t_out_vect + y_vect + z_vect)
            MH_b.append(-1 * tau_headway[(j1, j2, s, sp)])

        MH = np.array(MH)
        MH_b = np.array(MH_b)
        return MH, MH_b

    def _create_junction_condition_matrix(self, M, agv_routes, tau_operation):

        JC = []
        JC_b = []

        for y in self.y_iter:
            t_in_vect = [-1 if t == ("in", y[1], y[2]) else 0 for t in self.t_in_iter]
            t_out_vect = [1 if t == ("out", y[0], y[2]) else 0 for t in self.t_out_iter]
            y_vect = [-1 * M if yp == (y[1], y[0], y[2]) else 0 for yp in self.y_iter]
            z_vect = [0 for _ in self.z_iter]
            JC.append(t_in_vect + t_out_vect + y_vect + z_vect)
            JC_b.append(0)

        for j in self.J:
            for station in agv_routes[j]:
                t_in_vect = [1 if t == ("in", j, station) else 0 for t in self.t_in_iter]
                t_out_vect = [-1 if t == ("out", j, station) else 0 for t in self.t_out_iter]
                y_vect = [0 for _ in self.y_iter]
                z_vect = [0 for _ in self.z_iter]
                JC.append(t_in_vect + t_out_vect + y_vect + z_vect)
                JC_b.append(-1 * tau_operation[(j, station)])

        JC = np.array(JC)
        JC_b = np.array(JC_b)

        return JC, JC_b

    def _create_single_line_matrix(self, M):

        SL = []
        SL_b = []

        for j1, j2, s, sp in self.z_iter:
            t_in_vect = [1 if j_t == j1 and s_t == sp else 0 for _, j_t, s_t in self.t_in_iter]
            t_out_vect = [-1 if j_t == j2 and s_t == sp else 0 for _, j_t, s_t in self.t_out_iter]
            y_vect = [0 for _ in self.y_iter]
            z_vect = [-1 * M if zp == (j2, j1, sp, s) else 0 for zp in self.z_iter]
            SL.append(t_in_vect + t_out_vect + y_vect + z_vect)
            SL_b.append(0)

        SL = np.array(SL)
        SL_b = np.array(SL_b)

        return SL, SL_b

    def _create_bounds(self, d_max):

        t_in_min = [self.v_in[(j, s)] for _, j, s in self.t_in_iter]
        t_in_max = [self.v_in[(j, s)] + d_max[j] for _, j, s in self.t_in_iter]

        t_out_min = [self.v_out[(j, s)] for _, j, s in self.t_out_iter]
        t_out_max = [self.v_out[(j, s)] + d_max[j] for _, j, s in self.t_out_iter]

        y_min = [0 for _ in self.y_iter]
        y_max = [1 for _ in self.y_iter]

        z_min = [0 for _ in self.z_iter]
        z_max = [1 for _ in self.z_iter]

        x_min = t_in_min + t_out_min + y_min + z_min
        x_max = t_in_max + t_out_max + y_max + z_max

        bounds = [(x_min[i], x_max[i]) for i in range(len(self.x_iter))]
        return bounds



def print_ILP_size(A_ub, b_ub, A_eq, b_eq):
    print("ILP n.o. inequalities", len(b_ub))
    print("ILP, n.o. equalities", len(b_eq))
    print("ILP, n.o. vars", np.size(A_ub,1), "=", np.size(A_eq,1))
