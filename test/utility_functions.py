import unittest

import networkx as nx

from src import utils


class SingleStation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.M = 50
        cls.tracks = [("s0",)]
        cls.agv_routes = {0: ("s0",), 1: ("s0",)}

    def test_create_lists_single(self):
        stations = utils.create_stations_list(self.tracks)
        J = utils.create_agv_list(self.agv_routes)

        self.assertEqual(stations, ["s0"])  # add assertion here
        self.assertEqual(J, [0, 1])

    def test_create_graph_single(self):
        graph = utils.create_graph(self.tracks, self.agv_routes)
        self.assertEqual(graph.number_of_edges(), 0)
        self.assertEqual(graph.number_of_nodes(), 1)
        self.assertEqual(nx.get_node_attributes(graph, "pass_through")["s0"], [0, 1])


class MultipleStationsNoOpposite(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.M = 50
        cls.tracks = [("s0", "s1"), ("s1", "s0"), ("s0", "s2"), ("s2", "s3")]
        cls.agv_routes = {0: ("s0", "s1"), 1: ("s0", "s2", "s3")}

    def test_create_lists_multi(self):
        stations = utils.create_stations_list(self.tracks)
        J = utils.create_agv_list(self.agv_routes)

        self.assertEqual(set(stations), {"s0", "s1", "s2", "s3"})  # add assertion here
        self.assertEqual(J, [0, 1])

    def test_create_graph_multi(self):

        graph = utils.create_graph(self.tracks, self.agv_routes)
        self.assertEqual(graph.number_of_edges(), 4)
        self.assertEqual(graph.number_of_nodes(), 4)
        self.assertEqual(graph.number_of_edges("s0", "s1"), 2)

        self.assertEqual(nx.get_node_attributes(graph, "pass_through")["s0"], [0, 1])
        self.assertEqual(nx.get_node_attributes(graph, "pass_through")["s1"], [0])
        self.assertEqual(nx.get_node_attributes(graph, "pass_through")["s2"], [1])
        self.assertEqual(nx.get_node_attributes(graph, "pass_through")["s3"], [1])


if __name__ == '__main__':
    unittest.main()
