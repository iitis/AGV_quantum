import unittest
from src import LinearProg, qubo_solver, process_results
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
        self.p=3
        self.lp._to_bqm_qubo_ising(self.p)
        self.lp._to_cqm()
        self.opt = linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq, A_eq = lhs_eq, b_eq = rhs_eq, bounds = bnd, integrality=[1]*self.lp.nvars)
        self.x = self.opt.x
        self.obj = self.opt.fun


    def test_bqm_cqm(self):
        self.assertEqual(sorted(dimod.cqm_to_bqm(self.lp.cqm, self.p)[0].linear.values()), sorted(self.lp.bqm.linear.values()))
        self.assertEqual(sorted(dimod.cqm_to_bqm(self.lp.cqm, self.p)[0].quadratic.values()), sorted(self.lp.bqm.quadratic.values()))
        self.assertEqual(dimod.cqm_to_bqm(self.lp.cqm, self.p)[0].offset,self.lp.bqm.offset)


    def test_feasibility_check(self):
        """
        this test demenostrates check of feasilbity of the solution TODO Tomek we will use it for TN case
        """
        make_probabilistic_test = True
        bqm = self.lp.bqm
        sampleset = qubo_solver.sim_anneal(bqm, beta_range=(5, 100), num_sweeps=1000, num_reads=1000)
        # this is nost important line, it changes [0,1] encoding to linear
        sampleset = self.lp.interpreter(sampleset)

        prob=self.lp
        dict_list = []
        for data in sampleset.data():
            rdict = {}
            sample = data.sample
            rdict["energy"] = data.energy
            rdict["objective"] = round(process_results.get_objective(prob, sample), 2)
            rdict["feasible"] = all(process_results.analyze_constraints(prob, sample)[0].values())
            rdict["feas_constraints"] = process_results.analyze_constraints(prob, sample)
            dict_list.append(rdict)
        ret = sorted(dict_list, key=lambda d: d["energy"])[0]
        if make_probabilistic_test:
            assert ret["feasible"] == True
            assert ret["feas_constraints"][1] == 0 
        assert ret["objective"] < -9.

    def test_bqm_soln_sim(self):
        dict_list = qubo_solver.annealing(self.lp, "sim", "test_1", load=False, store=False)
        soln = (next((l for l in dict_list if l["feasible"]), None))
        assert soln["objective"] == self.obj
        assert (list(soln["sample"].values()) == self.x).all()


    def test_qubo(self):
        """
        this test compare bqm with qubo, we shall use it as well
        """

        vars = ['x_1[3]', 'eq_3_slack[3]', 'x_0[2]', 'eq_3_slack[4]', 'eq_2_slack[2]', 'eq_2_slack[4]', 'x_1[0]', 'eq_2_slack[1]', 'eq_1_slack[2]', 'x_0[0]', 'eq_3_slack[2]', 'x_0[1]', 'x_1[1]', 'eq_3_slack[0]', 'x_0[3]', 'x_1[2]', 'eq_1_slack[1]', 'eq_2_slack[0]', 'eq_2_slack[3]', 'eq_2_slack[5]', 'eq_1_slack[3]', 'eq_1_slack[4]', 'eq_1_slack[0]', 'eq_3_slack[1]']

        # variables
        for var in vars:
            assert var in self.lp.bqm.variables


        vars = self.lp.bqm.variables
        hs = self.lp.bqm.linear
        Js = self.lp.bqm.quadratic
        s = np.size(vars)

        count = 0
        ofset = 0
        for i in range(s):
            for j in range(i, s):
                if i == j:
                    if hs[vars[i]] != 0:
                        count = count + 1
                        assert hs[vars[i]] == self.lp.qubo[0][(vars[i], vars[i])]
                    else:
                        ofset = ofset + 1
                else:
                    if (vars[i], vars[j]) in Js:
                        J = Js[vars[i], vars[j]]
                        count = count + 1
                        try: 
                            assert J == self.lp.qubo[0][(vars[i], vars[j])]
                        except:
                            assert J == self.lp.qubo[0][(vars[j], vars[i])]


        assert count == len(self.lp.qubo[0])

        assert self.lp._count_linear_fields() + ofset == self.lp._count_qubits()

        assert count == self.lp._count_quadratic_couplings() + self.lp._count_linear_fields()

    def test_ising(self):

        assert len(self.lp.ising[0]) == self.lp._count_qubits()    # these are fields, 
        assert len(self.lp.ising[1]) == self.lp._count_quadratic_couplings()   # these are quadratic couplings
        self.lp.ising[2]  # this is energy ofset


        





