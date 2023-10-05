"""tests plotting train diagram and auciliary functions"""
import unittest
from AGV_quantum import plot_train_diagram, get_number_zones, zones_location, AGVS_coordinates



class TestTrainDiagram(unittest.TestCase):

    @classmethod

    def test_treain_diagram(self):
        assert 1 == 1

        tracks_len = {("s0", "s1"): 6, ("s1", "s0"): 6,
              ("s1", "s2"): 6, ("s2", "s1"): 6,
              }

        agv_routes = {0: ("s0", "s1"),
                    1: ("s1", "s2")}
        
        sol = {"t_in_0_s0": 0, "t_in_0_s1": 9, "t_in_1_s1": 7, "t_in_1_s2":15, "t_out_0_s0": 2, "t_out_0_s1": 11,
               "t_out_1_s1": 9, "t_out_1_s2": 17, "y_1_0_s1": 1
        }

        assert get_number_zones(tracks_len) == 3
        marks, zone_border = zones_location(tracks_len, n_zones=3, s_ofset=1)
        assert marks == {'s0': 0.5, 's1': 7.5, 's2': 14.5}
        assert zone_border == [0, 1, 7, 8, 14, 15]

        times, spaces = AGVS_coordinates(sol, agv_routes, marks, s_ofset=1)
        assert times == {0: [0, 2, 9, 11], 1: [7, 9, 15, 17]}
        assert spaces == {0: [0.0, 1.0, 7.0, 8.0], 1: [7.0, 8.0, 14.0, 15.0]}
        plot_train_diagram(sol, agv_routes, tracks_len)