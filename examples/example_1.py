from src import utils
from src.linear_solver import solve

M = 50
tracks = [("s0", "s1"), ("s1", "s0"),
          ("s1", "s2"), ("s2", "s1"),
          ("s2", "s3"), ("s3", "s2"),
          ("s3", "s4"), ("s4", "s3"),
          ("s4", "s5"), ("s5", "s4"),
          ("s5", "s6")]

# to include track_len into track would require some refactoring
tracks_len = {("s0", "s1"): 6, ("s1", "s0"): 6,
              ("s1", "s2"): 6, ("s2", "s1"): 6,
              ("s2", "s3"): 0, ("s3", "s2"): 0,
              ("s3", "s4"): 5, ("s4", "s3"): 5,
              ("s4", "s5"): 4, ("s5", "s4"): 4,
              ("s5", "s6"): 4, ("s6", "s5"): 4}

agv_routes = {0: ("s0", "s1", "s2", "s3"),
              1: ("s0", "s1", "s2"),
              2: ("s4", "s3", "s2", "s1"),
              3: ("s4", "s3", "s2", "s1", "s0"),
              4: ("s2", "s3"),
              5: ("s6", "s5", "s4", "s3"),
              6: ("s5", "s6")}

stations = utils.create_stations_list(tracks)
J = utils.create_agv_list(agv_routes)
agv_routes_as_edges = utils.agv_routes_as_edges(agv_routes)
all_same_way = utils.create_same_way_dict(agv_routes)

graph = utils.create_graph(tracks, agv_routes)

d_max = {i: 40 for i in J}
tau_pass = {(j, s, sp): tracks_len[(s, sp)] for j in J for s, sp in agv_routes_as_edges[j]}
tau_headway = {(j, jp, s, sp): 2 if (s, sp) != ("s2", "s3") and (s, sp) != ("s3", "s2") else 0
               for (j, jp) in all_same_way.keys() for (s, sp) in all_same_way[(j, jp)]}

tau_operation = {(agv, station): 2 for agv in J for station in stations}

initial_conditions = {("in", 0, "s0"): 0, ("in", 1, "s0"): 0, ("in", 2, "s4"): 8, ("in", 3, "s4"): 9,
                      ("in", 4, "s2"): 15, ("in", 5, "s6"): 0, ("in", 6, "s5"): 0}

weights = {j: 1 for j in J}

res, iterators = solve(M, tracks, tracks_len, agv_routes, d_max, tau_pass, tau_headway, tau_operation,
                    weights, initial_conditions)

if res.success:
    v_in, v_out = utils.create_v_in_out(tracks_len, agv_routes, tau_operation, iterators, initial_conditions)
    utils.nice_print(res, agv_routes, weights, d_max,  v_in, v_out, iterators)
else:
    print(res.message)
