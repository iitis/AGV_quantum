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

    def test_create_t_iter_single(self):
        in_out = "in"
        t_iter = utils.create_t_iterator(self.agv_routes, in_out)
        self.assertEqual(t_iter, [(in_out, 0, "s0"), (in_out, 1, "s0")])

    def test_create_y_iter_single(self):
        graph = utils.create_graph(self.tracks, self.agv_routes)
        y_iter = utils.create_y_iterator(graph)
        self.assertEqual(y_iter, [(0, 1, "s0"), (1, 0, "s0")])

    def test_create_z_iter_single(self):
        graph = utils.create_graph(self.tracks, self.agv_routes)
        z_iter = utils.create_z_iterator(graph, self.agv_routes)
        self.assertEqual(z_iter, [])


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

    def test_create_t_iter_multi(self):
        in_out = "in"
        t_iter = utils.create_t_iterator(self.agv_routes, in_out)
        self.assertEqual(set(t_iter), {(in_out, 0, "s0"), (in_out, 1, "s0"), (in_out, 0, "s1"),
                                       (in_out, 1, "s2"), (in_out, 1, "s3")})

    def test_create_y_iter_multi(self):
        graph = utils.create_graph(self.tracks, self.agv_routes)
        y_iter = utils.create_y_iterator(graph)
        self.assertEqual(y_iter, [(0, 1, "s0"), (1, 0, "s0")])

    def test_create_z_iter_multi(self):
        graph = utils.create_graph(self.tracks, self.agv_routes)
        z_iter = utils.create_z_iterator(graph, self.agv_routes)
        self.assertEqual(z_iter, [])


class TwoStationsOpposite(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.M = 50
        cls.tracks = [("s0", "s1"), ("s1", "s2"), ("s2", "s1")]
        cls.agv_routes = {0: ("s0", "s1", "s2"), 1: ("s1", "s0"), 2: ("s2", "s1")}

    def test_create_z_iter(self):
        graph = utils.create_graph(self.tracks, self.agv_routes)
        z_iter = utils.create_z_iterator(graph, self.agv_routes)
        self.assertEqual(z_iter, [(0, 1, "s0", "s1"), (1, 0, "s1", "s0")])


if __name__ == '__main__':
    unittest.main()
