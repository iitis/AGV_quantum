import unittest
import networkx as nx
from src import create_stations_list, create_agv_list, create_graph, create_t_iterator, create_y_iterator, create_z_iterator
from src import create_iterators, create_v_in_out


class SingleStation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.M = 50
        cls.tracks = [("s0",)]
        cls.agv_routes = {0: ("s0",), 1: ("s0",)}

    def test_create_lists_single(self):
        stations = create_stations_list(self.tracks)
        J = create_agv_list(self.agv_routes)

        self.assertEqual(stations, ["s0"])  # add assertion here
        self.assertEqual(J, [0, 1])

    def test_create_graph_single(self):
        graph = create_graph(self.tracks, self.agv_routes)
        self.assertEqual(graph.number_of_edges(), 0)
        self.assertEqual(graph.number_of_nodes(), 1)

        self.assertEqual(nx.get_node_attributes(graph, "pass_through")["s0"], [0, 1])

    def test_create_t_iter_single(self):
        in_out = "in"
        t_iter = create_t_iterator(self.agv_routes, in_out)
        self.assertEqual(t_iter, [(in_out, 0, "s0"), (in_out, 1, "s0")])

    def test_create_y_iter_single(self):
        graph = create_graph(self.tracks, self.agv_routes)
        y_iter = create_y_iterator(graph)
        self.assertEqual(y_iter, [(0, 1, "s0"), (1, 0, "s0")])

    def test_create_z_iter_single(self):
        graph = create_graph(self.tracks, self.agv_routes)
        z_iter = create_z_iterator(graph, self.agv_routes)
        self.assertEqual(z_iter, [])


class MultipleStationsNoOpposite(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.M = 50
        cls.tracks = [("s0", "s1"), ("s1", "s0"), ("s0", "s2"), ("s2", "s3")]
        cls.tracks_len = {("s0", "s1"): 1, ("s1", "s0"): 1,
                          ("s0", "s2"): 1, ("s2", "s0"): 1,
                          ("s2", "s3"): 1, ("s3", "s2"): 1}

        cls.agv_routes = {0: ("s0", "s1"), 1: ("s0", "s2", "s3")}
        cls.graph = create_graph(cls.tracks, cls.agv_routes)
        cls.iterators = create_iterators(cls.graph, cls.agv_routes)
        cls.d_max = {j: 10 for j in cls.agv_routes.keys()}
        cls.initial_conditions = {("in", 0, "s0"): 0, ("in", 1, "s0"): 2}

        cls.J = create_agv_list(cls.agv_routes)
        cls.stations = create_stations_list(cls.tracks)
        cls.tau_operation = {(agv, station): 2 for agv in cls.J for station in cls.stations}

    def test_create_lists_multi(self):
        stations = create_stations_list(self.tracks)
        J = create_agv_list(self.agv_routes)

        self.assertEqual(set(stations), {"s0", "s1", "s2", "s3"})  # add assertion here
        self.assertEqual(J, [0, 1])

    def test_create_graph_multi(self):

        graph = create_graph(self.tracks, self.agv_routes)
        self.assertEqual(graph.number_of_edges(), 4)
        self.assertEqual(graph.number_of_nodes(), 4)
        self.assertEqual(graph.number_of_edges("s0", "s1"), 2)

        self.assertEqual(nx.get_node_attributes(graph, "pass_through")["s0"], [0, 1])
        self.assertEqual(nx.get_node_attributes(graph, "pass_through")["s1"], [0])
        self.assertEqual(nx.get_node_attributes(graph, "pass_through")["s2"], [1])
        self.assertEqual(nx.get_node_attributes(graph, "pass_through")["s3"], [1])

    def test_create_t_iter_multi(self):
        in_out = "in"
        t_iter = create_t_iterator(self.agv_routes, in_out)
        self.assertEqual(set(t_iter), {(in_out, 0, "s0"), (in_out, 1, "s0"), (in_out, 0, "s1"),
                                       (in_out, 1, "s2"), (in_out, 1, "s3")})

    def test_create_y_iter_multi(self):
        graph = create_graph(self.tracks, self.agv_routes)
        y_iter = create_y_iterator(graph)
        self.assertEqual(y_iter, [(0, 1, "s0"), (1, 0, "s0")])

    def test_create_z_iter_multi(self):
        graph = create_graph(self.tracks, self.agv_routes)
        z_iter = create_z_iterator(graph, self.agv_routes)
        self.assertEqual(z_iter, [])

    def test_v_in_out(self):
        v_in, v_out = create_v_in_out(self.tracks_len, self.agv_routes, self.tau_operation,
                                            self.iterators, self.initial_conditions)
        self.assertEqual(v_in, {(0, "s0"): 0, (0, "s1"): 3, (1, "s0"): 2, (1, "s2"): 5, (1, "s3"): 8})
        self.assertEqual(v_out, {(0, 's0'): 2, (0, 's1'): 5, (1, 's0'): 4, (1, 's2'): 7, (1, 's3'): 10})


class TwoStationsOpposite(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.M = 50
        cls.tracks = [("s0", "s1"), ("s1", "s2"), ("s2", "s1")]
        cls.agv_routes = {0: ("s0", "s1", "s2"), 1: ("s1", "s0"), 2: ("s2", "s1")}

    def test_create_z_iter(self):
        graph = create_graph(self.tracks, self.agv_routes)
        z_iter = create_z_iterator(graph, self.agv_routes)
        self.assertEqual(z_iter, [(0, 1, "s0", "s1"), (1, 0, "s1", "s0")])


if __name__ == '__main__':
    unittest.main()
