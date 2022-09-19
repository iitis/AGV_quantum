from src import utils
from src.linear_solver import solve

M = 50
tracks = [("s0",)]
agv_routes = {0: ("s0",), 1: ("s0",)}

stations = utils.create_stations_list(tracks)
J = utils.create_agv_list(agv_routes)

d_max = {i: 5 for i in J}
#tau_pass = {(agv, way[0], way[1]): 5 for agv, way in agv_routes.items()}
#tau_headway = {}
#tau_operation = {(agv, station): 1 for agv in J for station in stations}

graph = utils.create_graph(tracks, stations, agv_routes)
t_iter = utils.create_t_iterator(agv_routes, "out")
print(t_iter)

#res, x_iter = solve(M, tracks, agv_routes, d_max, tau_pass, tau_headway, tau_operation,
#                    weights={0: 1, 1: 1}, initial_conditions={})
#print(utils.see_variables(res.x, x_iter))
