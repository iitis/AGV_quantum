# 15 AGVs 7 zones d_max = 40

from AGV_quantum import create_stations_list, create_agv_list, create_graph, create_same_way_dict, agv_routes_as_edges


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
              1: ("s0", "s1", "s2", "s3"),
              2: ("s0", "s1", "s2", "s3"),
              3: ("s0", "s1", "s2", "s3"),
              4: ("s0", "s1", "s2"),
              5: ("s0", "s1", "s2"),
              6: ("s4", "s3", "s2", "s1"),
              7: ("s4", "s3", "s2", "s1", "s0"),
              8: ("s4", "s3", "s2", "s1", "s0"),
              9: ("s4", "s3", "s2", "s1", "s0"),
              10: ("s4", "s3", "s2", "s1", "s0"),
              11: ("s4", "s3", "s2", "s1", "s0"),
              12: ("s2", "s3"),
              13: ("s6", "s5", "s4", "s3"),
              14: ("s5", "s6"),
              15: ("s4", "s3", "s2", "s1", "s0"),
              16: ("s4", "s3", "s2", "s1", "s0"),
              17: ("s4", "s3", "s2", "s1", "s0"),
              18: ("s2", "s3", "s4"),
              19: ("s6", "s5", "s4", "s3", "s2"),
              20: ("s5", "s6")
              }

stations = create_stations_list(tracks)
J = create_agv_list(agv_routes)
agv_routes_as_e = agv_routes_as_edges(agv_routes)
all_same_way = create_same_way_dict(agv_routes)
graph = create_graph(tracks, agv_routes)

d_max = {i: 40 for i in J}
tau_pass = {(j, s, sp): tracks_len[(s, sp)] for j in J for s, sp in agv_routes_as_e[j]}
tau_headway = {(j, jp, s, sp): 2 if (s, sp) != ("s2", "s3") and (s, sp) != ("s3", "s2") else 0
               for (j, jp) in all_same_way for (s, sp) in all_same_way[(j, jp)]}

tau_operation = {(agv, station): 2 for agv in J for station in stations}

initial_conditions = {("in", 0, "s0"): 0, ("in", 1, "s0"): 1, ("in", 2, "s0"): 2, ("in", 3, "s0"): 3,
                      ("in", 4, "s0"): 4, ("in", 5, "s0"): 5, ("in", 6, "s4"): 0, ("in", 7, "s4"): 1,
                      ("in", 8, "s4"): 2, ("in", 9, "s4"): 3, ("in", 10, "s4"): 8,
                      ("in", 11, "s4"): 5,
                      ("in", 12, "s2"): 7, ("in", 13, "s6"): 9, ("in", 14, "s5"): 9,
                      ("in", 15, "s4"): 4, ("in", 16, "s4"): 7,
                      ("in", 17, "s4"): 7,
                      ("in", 18, "s2"): 8,
                      ("in", 19, "s6"): 10,
                      ("in", 20, "s5"): 10
                    }

weights = {j: 1 for j in J}

