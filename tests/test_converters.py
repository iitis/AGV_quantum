import unittest
from src import LinearProg, qubo_solver
import dimod
from scipy.optimize import linprog
import numpy as np

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
        self.lp._to_bqm_and_qubo(self.p)
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


    def test_qubo(self):
        vars = ['x_1[3]', 'eq_3_slack[3]', 'x_0[2]', 'eq_3_slack[4]', 'eq_2_slack[2]', 'eq_2_slack[4]', 'x_1[0]', 'eq_2_slack[1]', 'eq_1_slack[2]', 'x_0[0]', 'eq_3_slack[2]', 'x_0[1]', 'x_1[1]', 'eq_3_slack[0]', 'x_0[3]', 'x_1[2]', 'eq_1_slack[1]', 'eq_2_slack[0]', 'eq_2_slack[3]', 'eq_2_slack[5]', 'eq_1_slack[3]', 'eq_1_slack[4]', 'eq_1_slack[0]', 'eq_3_slack[1]']

        # variables
        for var in vars:
            assert var in self.lp.bqm.variables


        vars = self.lp.bqm.variables
        hs = self.lp.bqm.linear
        Js = self.lp.bqm.quadratic
        s = np.size(vars)

        count = 0
        for i in range(s):
            for j in range(i, s):
                if i == j:
                    if hs[vars[i]] != 0:
                        assert hs[vars[i]] == self.lp.qubo[0][(vars[i], vars[i])]
                        count = count + 1
                else:
                    try:
                        J = Js[vars[i], vars[j]]
                        count = count + 1
                    except:
                        J = 0 
                    if J != 0:
                        try: 
                            assert J == self.lp.qubo[0][(vars[i], vars[j])]
                        except:
                            assert J == self.lp.qubo[0][(vars[j], vars[i])]

        assert count == len(self.lp.qubo[0])
        





