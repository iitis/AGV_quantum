import unittest
from src import LinearProg, qubo_solver
import dimod
from scipy.optimize import linprog

class BQMConverter(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        obj = [-1, -2]
        lhs_ineq = [[2, 1],[-4, 5], [1, -2]]
        rhs_ineq = [20, 10, 2]
        lhs_eq = [[-1, 5]]
        rhs_eq = [15]
        bnd = [(0, 8), (0, 10)]
        self.lp = LinearProg.LinearProg(c=obj, bounds=bnd, A_ub=lhs_ineq, b_ub=rhs_ineq, A_eq=lhs_eq, b_eq=rhs_eq)
        self.p=2
        self.lp._to_bqm(self.p)
        self.lp._to_cqm()
        self.opt = linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq, A_eq = lhs_eq, b_eq = rhs_eq, bounds = bnd, integrality=[1]*self.lp.nvars)
        self.x = self.opt.x
        self.obj = self.opt.fun


    def test_bqm_cqm(self):
        self.assertEqual(sorted(dimod.cqm_to_bqm(self.lp.cqm, self.p)[0].linear.values()), sorted(self.lp.bqm.linear.values()))
        self.assertEqual(sorted(dimod.cqm_to_bqm(self.lp.cqm, self.p)[0].quadratic.values()), sorted(self.lp.bqm.quadratic.values()))
        self.assertEqual(dimod.cqm_to_bqm(self.lp.cqm, self.p)[0].offset,self.lp.bqm.offset)

    def test_bqm_soln_sim(self):
        dict_list = qubo_solver.annealing(self.lp, "sim", "test_1", load=False, store=False)
        soln = (next((l for l in dict_list if l["feasible"]), None))
        assert soln["objective"] == self.obj
        assert (list(soln["sample"].values()) == self.x).all()
