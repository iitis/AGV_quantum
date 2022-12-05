from src.quadratic_solver import load_linear_prog_object
from src.utils import qubo_to_matrix
import scipy.sparse as sparse

if __name__ == "__main__":
    p = 2.75
    lp = load_linear_prog_object("lp_files/lp_largest.pkl")
    lp._to_bqm_qubo_ising(p)
    qubo = lp.qubo[0]
    matrix = qubo_to_matrix(qubo, lp)
    matrix = sparse.coo_matrix(matrix)
    sparse.save_npz("largest_qubo_coo.npz", matrix)
