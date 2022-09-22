from src import utils
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


    res, iterators = solve(M, tracks, agv_routes, d_max, tau_pass, tau_headway, tau_operation,
                        weights={0: 1, 1: 1/2}, initial_conditions={})

    if res.success:
        sol = utils.see_variables(res.x, iterators["x"])
        utils.nice_print(sol, agv_routes, iterators)
    else:
        print(res.message)

if __name__ == "__main__":
    test_one_station()
