from src import utils
from src.linear_solver import make_linear_problem
from src.linear_solver import solve


def test_one_station():
    """
        s0                           
    1 ->    ->
    0 ->    ->
    """


    M = 50
    tracks = [("s0",)]
    agv_routes = {0: ("s0",), 1: ("s0",)}

    stations = utils.create_stations_list(tracks)
    J = utils.create_agv_list(agv_routes)

    d_max = {i: 5 for i in J}
    tau_pass = {}
    tau_headway = {}
    tau_operation = {(agv, station): 1 for agv in J for station in stations}

    weights = {0: 1, 1: 1 / 2}
    initial_conditions = {}
    tracks_len = {}

    c, A_ub, b_ub, A_eq, b_eq, bounds, iterators = make_linear_problem(M, tracks, {}, agv_routes, d_max, tau_pass, tau_headway, tau_operation,
                           weights, initial_conditions)

    res, iterators = solve(c, A_ub, b_ub, A_eq, b_eq, bounds, iterators)
    assert res.success

if __name__ == "__main__":
    test_one_station()
