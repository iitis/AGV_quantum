from quadratic_solver import load_linear_prog_object, count_vertices
import scipy.sparse as sparse
import pandas as pd
import numpy as np


def qubo_to_matrix(qubo: dict) -> np.ndarray:
    qubo = dict(sorted(qubo.items()))

    data = sorted(list(lp.bqm.variables))
    df = pd.DataFrame(columns=data, index=data)

    for item, value in qubo.items():
        df.at[item[0], item[1]] = value
    df.fillna(0, inplace=True)
    array = df.to_numpy()
    return array


if __name__ == "__main__":
    lp = load_linear_prog_object("../lp_tiny.pkl")
    qubo = lp.qubo[0]
    matrix = qubo_to_matrix(qubo)
    matrix = sparse.coo_matrix(matrix)
    sparse.save_npz("tiny_qubo_coo.npz", matrix)
