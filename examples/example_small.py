# 4 AGV 5 zoneas d_max = 10 example

from AGV_quantum import create_stations_list, create_agv_list, create_graph, create_same_way_dict, agv_routes_as_edges


M = 28
tracks = [("s0", "s1"), ("s1", "s0"),
          ("s1", "s2"), ("s2", "s1"),
          ("s2", "s3"), ("s3", "s2"),
          ("s3", "s4"), ("s4", "s3")]

# to include track_len into track would require some refactoring
tracks_len = {("s0", "s1"): 6, ("s1", "s0"): 6,
              ("s1", "s2"): 6, ("s2", "s1"): 6,
              ("s2", "s3"): 0, ("s3", "s2"): 0,
              ("s3", "s4"): 5, ("s4", "s3"): 5
              }

agv_routes = {0: ("s0", "s1", "s2"),
              1: ("s1", "s2"),
              2: ("s4", "s3", "s2"),
              4: ("s2", "s3")
              }

stations = create_stations_list(tracks)
J = create_agv_list(agv_routes)
agv_routes_as_e = agv_routes_as_edges(agv_routes)
all_same_way = create_same_way_dict(agv_routes)

graph = create_graph(tracks, agv_routes)

d_max = {i: 10 for i in J}
tau_pass = {(j, s, sp): tracks_len[(s, sp)] for j in J for s, sp in agv_routes_as_e[j]}
tau_headway = {(j, jp, s, sp): 2 if (s, sp) != ("s2", "s3") and (s, sp) != ("s3", "s2") else 0
               for (j, jp) in all_same_way for (s, sp) in all_same_way[(j, jp)]}

tau_operation = {(agv, station): 2 for agv in J for station in stations}

initial_conditions = {("in", 0, "s0"): 0, ("in", 1, "s1"): 8, ("in", 2, "s4"): 8, ("in", 4, "s2"): 15}

weights = {j: 1 for j in J}
