import unittest
import pickle
from dwave.embedding import embed_qubo, diagnose_embedding
import dwave_networkx as dnx
import minorminer
import networkx as nx
from src.embed_to_SpinGlassEngine import add_zero_h_qubo, add_zero_h_ising, load_linear_prog_object, compute_edges, \
    compute_vertices, create_qubo_graph, create_ising_graph


class TestEmbedder(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.lp = load_linear_prog_object("lp.pkl")
        cls.chimera = dnx.chimera_graph(16, 16)

    def test_qubo(self):
        self.lp = add_zero_h_qubo(self.lp)
        s1 = compute_vertices(self.lp.qubo[0])
        self.assertEqual(s1, len(self.lp.bqm.variables))

        s2 = compute_edges(self.lp.qubo[0])
        self.assertEqual(s2, len(self.lp.bqm.quadratic.values()))

    def test_qubo_graph(self):
        g = create_qubo_graph(self.lp)
        self.assertEqual(g.number_of_nodes(), len(self.lp.bqm.variables))
        self.assertEqual(g.number_of_edges(), len(self.lp.bqm.quadratic.values()))

    def test_ising_graph(self):
        g = create_ising_graph(self.lp)
        self.assertEqual(g.number_of_nodes(), len(self.lp.bqm.variables))
        self.assertEqual(g.number_of_edges(), len(self.lp.bqm.quadratic.values()))

    def test_ising(self):
        self.lp = add_zero_h_ising(self.lp)

        self.assertEqual(len(self.lp.ising[0]), len(self.lp.bqm.variables))
        self.assertEqual(len(self.lp.ising[1]), len(self.lp.bqm.quadratic.values()))


if __name__ == '__main__':
    unittest.main()
