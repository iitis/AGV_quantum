from src.process_results import load_results
from src.LinearProg import LinearProg
import os
import pickle
import json
from dwave.embedding import embed_qubo, diagnose_embedding, embed_ising
import dwave_networkx as dnx
import minorminer
import networkx as nx

"""
Right now rather dirty. TODO: better design
with open("../examples/qubo.pkl", "rb") as f:
    qubo = pickle.load(f)
"""

#TODO rewrite for ising
def add_zero_h(lp: LinearProg) -> LinearProg:
    for var in lp.bqm.variables:
        if (var, var) not in lp.qubo[0]:
            lp.qubo[0][(var, var)] = 0
    return lp


def load_linear_prog_object(lp_location: str) -> LinearProg:
    with open(lp_location, "rb") as f:
        lp = pickle.load(f)
    p = 2.75
    lp._to_bqm_qubo_ising(p)
    return lp


def create_qubo_graph(lp: LinearProg) -> nx.Graph:
    edges = []
    g = nx.Graph()
    for key in lp.qubo[0].keys():
        if key[0] != key[1]:
            edges.append(key)
    g.add_edges_from(edges)
    return g


def find_embedding(source: nx.Graph, target: nx.Graph) -> dict:
    print("searching for graph embedding.....")
    embedding = minorminer.find_embedding(source, target, verbose=1)
    print("graph embedding found")
    return embedding


def compute_vertices(qubo):
    s = 0
    for key in qubo.keys():
        if key[0] == key[1]:
            s += 1
    return s


def compute_edges(qubo: dict) -> int:
    s = 0
    for key in qubo.keys():
        if key[0] != key[1]:
            s += 1
    return s

if __name__ == "__main__":
    lp = load_linear_prog_object("../lp.pkl")
    lp = add_zero_h(lp)
    chimera = dnx.chimera_graph(16)
    ising_graph = create_qubo_graph(lp)
    embedding = find_embedding(ising_graph, chimera)

    diagnosis = diagnose_embedding(embedding, ising_graph, chimera)
    for problem in diagnosis:
        print(problem)


    print("embedding QUBO")
    final_qubo = embed_ising(lp.qubo[0], embedding, chimera)
    print("done")

    print("embedded qubo vertices: ", compute_vertices(final_qubo))
    print("embedded qubo edges: ", compute_edges(final_qubo))
    with open("embedded_qubo.txt", "w") as f:
        for spin in sorted(chimera.nodes):
            if (spin, spin) in final_qubo:
                f.write(f"{spin + 1} {spin + 1} {final_qubo[(spin, spin)]}\n")
            else:
                f.write(f"{spin + 1} {spin + 1} {0}\n")
        for edge in chimera.edges:
            if edge in final_qubo:
                f.write(f"{edge[0] + 1} {edge[1] + 1} {final_qubo[edge]}\n")
            else:
                pass
                f.write(f"{edge[0] + 1} {edge[1] + 1} {0}\n")

        """for key, value in sorted(final_qubo.items()):
            f.write(f"{key[0] + 1} {key[1] + 1} {value} \n")"""
