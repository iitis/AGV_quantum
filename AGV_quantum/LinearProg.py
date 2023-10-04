import dimod
from cpp_pyqubo import Binary, Constraint, Placeholder
from pyqubo import LogEncInteger
from pyqubo import Binary



class LinearProg:
    def __init__(self, c, bounds, A_ub=None, b_ub=None, A_eq=None, b_eq=None):
        assert not (A_ub is None and A_eq is None), "Both can not be None."
        assert not (A_ub is not None and b_ub is None), "Enter b_ub."
        assert not (A_ub is None and b_ub is not None), "Enter A_ub."
        assert not (A_eq is not None and b_eq is None), "Enter b_eq."
        assert not (A_eq is None and b_eq is not None), "Enter b_eq."
        self.c = c
        self.A_ub = A_ub
        self.b_ub = b_ub
        self.A_eq = A_eq
        self.b_eq = b_eq
        self.bounds = bounds
        self.nvars = 0
        self.num_eq = 0
        self.var_names = []
        self.set_vars()
        self.bqm = None
        self.interpreter = None
        self.cqm = None
        self.Q = None

    def set_vars(self):
        """Sets the number of variables and variable names"""
        self.nvars = len(self.bounds)
        self.var_names = [f"x_{i}" for i in range(self.nvars)]

    def _to_bqm_qubo_ising(self, pdict=None):
        """Converts linear program into binary quadratic model

        :param pdict: Dictionary of penalties
        :type pdict: dict
        """
        c, A_ub, b_ub, A_eq, b_eq, bounds = (
            self.c,
            self.A_ub,
            self.b_ub,
            self.A_eq,
            self.b_eq,
            self.bounds,
        )
        H = 0
        ind = 0
        vars = []
        for (lb, ub) in self.bounds:
            if lb == 0 and ub == 1:
                vars.append(Binary(f"x_{ind}"))
            else:
                vars.append(LogEncInteger(f"x_{ind}", (lb, ub)))
            ind += 1

        pyqubo_obj = sum(var * coef for var, coef in zip(vars, c) if coef != 0)
        H += Placeholder("obj") * pyqubo_obj

        num_eq = 0
        if A_eq is not None:
            for i in range(len(A_eq)):
                expr = sum(
                    A_eq[i][j] * vars[j] for j in range(self.nvars) if A_eq[i][j] != 0
                )
                expr -= b_eq[i]
                H += Constraint(Placeholder(f"eq_{num_eq}") * expr ** 2, f"eq_{num_eq}")
                num_eq += 1
        if A_ub is not None:
            for i in range(len(A_ub)):
                expr = sum(
                    A_ub[i][j] * vars[j] for j in range(self.nvars) if A_ub[i][j] != 0
                )
                expr -= b_ub[i]
                slack = LogEncInteger(
                    f"eq_{num_eq}_slack",
                    (0, LinearProg._get_slack_ub(vars, A_ub[i], b_ub[i])),
                )

                H += Constraint(
                    Placeholder(f"eq_{num_eq}") * (expr + slack) ** 2,
                    f"eq_{num_eq}",
                )

                num_eq += 1

        self.num_eq = num_eq
        pyqubo_model = H.compile()
        if pdict == None:
            pdict = {f"eq_{i}": 2 for i in range(self.num_eq)}
        elif type(pdict) == int or type(pdict) == float:
            pdict = {f"eq_{i}": pdict for i in range(self.num_eq)}
        pdict["obj"] = 1
        self.qubo = pyqubo_model.to_qubo(feed_dict=pdict)
        self.ising = pyqubo_model.to_ising(feed_dict=pdict)
        self.bqm = pyqubo_model.to_bqm(feed_dict=pdict)

        def interpreter(sampleset: dimod.SampleSet):
            """This is an interpreter function for binary quadratic model. It decodes the binary variables in the sample back to integer variables using pyqubo

            :param sampleset: Sampleset to analyze
            :type sampleset: dimod.SampleSet
            :return: New sampleset with integer variables
            :rtype: dimod.SampleSet
            """
            result = []
            energies = [d.energy for d in sampleset.data()]
            for sample in sampleset.samples():
                decoded = pyqubo_model.decode_sample(
                    dict(sample), vartype="BINARY", feed_dict=pdict
                )

                decoded_dict = {**decoded.subh, **decoded.sample}

                result.append({v: int(decoded_dict[v]) for v in map(str, self.var_names)})

            return dimod.SampleSet.from_samples(
                dimod.as_samples(result), "INTEGER", energy=energies)


        self.interpreter = lambda ss: interpreter(ss)

    @staticmethod
    def _get_slack_ub(vars: list, coefs: list, offset: int) -> int:
        """Returns upper bound for slack variables

        :param vars: List of variables (can be integer or binary)
        :type vars: list
        :param coefs: List of coefficients for the inequality
        :type coefs: list
        :param offset: RHS of the inequality
        :type offset: int
        :return: Upper bound for the slack variable
        :rtype: int
        """
        ub = 0
        for var, coef in zip(vars, coefs):
            if type(var) == LogEncInteger:
                ub += coef * (var.value_range[1] if coef < 0 else var.value_range[0])
            else:
                ub += coef * (1 if coef < 0 else 0)

        result = -ub + offset
        assert int(result) == result
        return int(result)

    def _to_cqm(self):
        """Converts linear program into constrained quadratic model"""
        c, A_ub, b_ub, A_eq, b_eq, bounds = (
            self.c,
            self.A_ub,
            self.b_ub,
            self.A_eq,
            self.b_eq,
            self.bounds,
        )

        ind = 0
        vars = []
        for (lb, ub) in self.bounds:
            if lb == 0 and ub == 1:
                vars.append(dimod.Binary(f"x_{ind}"))
            else:
                vars.append(dimod.Integer(f"x_{ind}", lower_bound=lb, upper_bound=ub))
            ind += 1

        cqm = dimod.ConstrainedQuadraticModel()
        dimod_obj = sum(var * coef for var, coef in zip(vars, c) if coef != 0)
        cqm.set_objective(dimod_obj)

        num_eq = 0
        if A_eq is not None:
            for i in range(len(A_eq)):
                expr = sum(
                    A_eq[i][j] * vars[j] for j in range(self.nvars) if A_eq[i][j] != 0
                )
                new_c = expr == b_eq[i]
                cqm.add_constraint(new_c, label=f"eq_{num_eq}")
                num_eq += 1
        if A_ub is not None:
            for i in range(len(A_ub)):
                expr = sum(
                    A_ub[i][j] * vars[j] for j in range(self.nvars) if A_ub[i][j] != 0
                )
                new_c = expr <= b_ub[i]
                cqm.add_constraint(new_c, label=f"eq_{num_eq}")
                num_eq += 1
        self.num_eq = num_eq
        self.cqm = cqm

    def _count_qubits(self):
        vars = self.bqm.variables
        return len(vars) 


    def _count_quadratic_couplings(self):
        """
        returns number of copulings - Js
        """
        count = 0
        for J in self.bqm.quadratic.values():
            if J != 0:
                count = count + 1
        return count


    def _count_linear_fields(self):
        """
        return number of local fields hs
        """ 
        count = 0
        for h in self.bqm.linear.values():
            if h != 0:
                count = count + 1
        return count             

    

