import unittest
import pickle
from dwave.embedding import embed_qubo, diagnose_embedding
import dwave_networkx as dnx
import minorminer
import networkx as nx
from src.embed_to_SpinGlassEngine import add_zero_h, load_linear_prog_object, create_qubo_graph


class TestEmbedder(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.lp = load_linear_prog_object("lp.pkl")
        cls.lp = add_zero_h(cls.lp)

        cls.chimera = dnx.chimera_graph(16, 16)

    def test_qubo(self):
        s = 0
        for key in self.lp.qubo[0].keys():
            if key[0] == key[1]:
                s += 1
        self.assertEqual(s, len(self.lp.bqm.variables))

        s = 0
        for key in self.lp.qubo[0].keys():
            if key[0] != key[1]:
                s += 1
        self.assertEqual(s, len(self.lp.bqm.quadratic.values()))

    def test_qubo_graph(self):
        g = create_qubo_graph(self.lp)
        self.assertEqual(g.number_of_nodes(), len(self.lp.bqm.variables))
        self.assertEqual(g.number_of_edges(), len(self.lp.bqm.quadratic.values()))


if __name__ == '__main__':
    unittest.main()
