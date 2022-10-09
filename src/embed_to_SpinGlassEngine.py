from src.process_results import load_results
from src.LinearProg import LinearProg
import os
import pickle
import json
from dwave.embedding import embed_qubo, diagnose_embedding, embed_ising
import dwave_networkx as dnx
import minorminer
import networkx as nx
import pandas as pd
"""
Right now rather dirty. TODO: better design
with open("../examples/qubo.pkl", "rb") as f:
    qubo = pickle.load(f)
"""

# TODO rewrite for ising


def add_zero_h_qubo(lp: LinearProg) -> LinearProg:
    for var in lp.bqm.variables:
        if (var, var) not in lp.qubo[0]:
            lp.qubo[0][(var, var)] = 0
    return lp


def add_zero_h_ising(lp: LinearProg) -> LinearProg:
    for var in lp.bqm.variables:
        if var not in lp.ising[0]:
            lp.ising[0][var] = 0
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


def create_ising_graph(lp: LinearProg) -> nx.Graph:
    edges = lp.ising[1].keys()
    g = nx.Graph()
    g.add_edges_from(edges)
    return g


def find_embedding(source: nx.Graph, target: nx.Graph) -> dict:
    print("searching for graph embedding.....")
    embedding = minorminer.find_embedding(source, target, verbose=1)
    print("graph embedding found")
    return embedding


def compute_vertices(qubo: dict) -> int:
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

    lp = load_linear_prog_object("../lp_big.pkl")
    lp = add_zero_h_ising(lp)
    chimera = dnx.chimera_graph(32)
    print(chimera.number_of_edges())
    ising_graph = create_ising_graph(lp)

    embedding = find_embedding(ising_graph, chimera)

    diagnosis = diagnose_embedding(embedding, ising_graph, chimera)
    for problem in diagnosis:
        print(problem)

    """results = pd.read_csv("../agv_results.csv", delimiter=";", header=None)
    result_data = results.iat[0, 2]
    result_data = result_data.replace("(", "{")
    result_data = result_data.replace(")", "}")
    result_data = result_data.replace("=>", ":")
    result_data = result_data.replace("Dict", "")
    result_dict = eval(result_data)
    


  
    lp = load_linear_prog_object("../lp.pkl")
    lp = add_zero_h_ising(lp)
    chimera = dnx.chimera_graph(16)
    ising_graph = create_ising_graph(lp)

    embedding = find_embedding(ising_graph, chimera)

    diagnosis = diagnose_embedding(embedding, ising_graph, chimera)
    for problem in diagnosis:
        print(problem)

    print("embedding ising")
    final_ising = embed_ising(lp.ising[0], lp.ising[1], embedding, chimera)
    print("done")

    max_value = max(max(lp.ising[0].values()), max(lp.ising[1].values()))
    min_value = min(min(lp.ising[0].values()), min(lp.ising[1].values()))
    renom = max(abs(max_value), abs(min_value))
    with open("embedded_ising.txt", "w") as f:
        for spin in sorted(chimera.nodes):
            if spin in final_ising[0]:
                f.write(f"{spin + 1} {spin + 1} {final_ising[0][spin]/renom}\n")
            else:
                f.write(f"{spin + 1} {spin + 1} {0}\n")
        for edge in chimera.edges:
            if edge in final_ising[1]:
                f.write(f"{edge[0] + 1} {edge[1] + 1} {final_ising[1][edge]/renom}\n")
            else:
                f.write(f"{edge[0] + 1} {edge[1] + 1} {0}\n")"""
