from src import utils
from src.linear_solver import solve


def test_line_fragemnt():
    """
        s0                           s1
    1 -> 
    0 ->  --------------------------
    """


    M = 50
    tracks = [("s0", "s1"), ("s1", "s0")]
    agv_routes = {0: ("s0", "s1"), 1: ("s0", "s1")}
    weights = {0: 1, 1: 1}

    stations = utils.create_stations_list(tracks)
    J = utils.create_agv_list(agv_routes)

    d_max = {i: 5 for i in J}
    tau_pass = {(agv, way[0], way[1]): 5 for agv, way in agv_routes.items()}
    tau_headway = {(0, 1, "s0", "s1"): 2, (1, 0, "s0", "s1"): 2}
    tau_operation = {(agv, station): 1 for agv in J for station in stations}

    res, x_iter = solve(M, tracks, agv_routes, d_max, tau_pass, tau_headway, tau_operation,
                        weights)

    if res.success:
        print(utils.see_variables(res.x, x_iter))
    else:
        print(res.message)
