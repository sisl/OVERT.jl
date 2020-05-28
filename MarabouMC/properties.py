# property classes
from MC_constraints import Constraint, ConstraintType, MatrixConstraint, ReluConstraint, Monomial, MaxConstraint
import copy

# property
class Property():
    """
    Property that you want to hold.
    For now, box set over state variables.
    Future: linear constraints on outputs OR box set.
    """
    def __init__(self, arg):
        pass

    def complement(self):
        # return complement of desired property
        raise NotImplementedError

# property
class ConstraintProperty():
    """
    Property that you want to hold. A list of constraints in CNF.
    NOTE: For now, all properties must have constraints with strict inequalities,
    so that when they are inverted this yields non-strict inequalities.
    """
    def __init__(self, c, outputs):
        self.constraints = c
        self.constraint_complements = []
        self.outputs = outputs # list of the variables we are asserting constraints over
        self.next_new_var = 1
    
    def __repr__(self):
        out = "<ConstraintProperty: "
        out += str(self.n_constraints()) + " constraints >"
        for c in self.constraints:
            out += "\n" + c.__repr__() + " && "
        out = out[:-4]
        out += "\n"
        return out
    
    def n_constraints(self):
        return len(self.constraints)
    
    def complement(self):
        """
        Get complement of property in CNF.
        """
        # first get in DNF
        DNF_complements = self.complements_DNF()
        
        # convert to CNF
        CNF_complements = self.convert_to_CNF(DNF_complements)
        return CNF_complements
    
    def get_new_var(self):
        nnv = "Y"+str(self.next_new_var)
        self.next_new_var += 1
        return nnv

    def complements_DNF(self):
        """
        Get complement of property in DNF.
        """
        complements = []
        for c in self.constraints:
            ccomp = c.complement()
            if isinstance(ccomp, list):
                complements.extend(ccomp)
            else:
                complements.append(ccomp)
        return complements

    def cnf_conversion_helper(self, constraint):
        """
        Take a constraint of the form 
        5x + 6y -5 R 0 and turns it into

        Y == 5x + 6y - 5
        (for later to assert: Y >=0 or max(Y, somethingelse) >= 0)
        """
        # turn into >= inequality: -5x -6y <= -5  -->  5x + 6y >= 5
        geq_comp = constraint.get_geq()
        # define new var: Y = 5x + 6y - 5
        new_var_constraint = copy.deepcopy(geq_comp)
        new_var_constraint.type = ConstraintType('EQUALITY')
        Y = self.get_new_var()
        new_var_constraint.monomials += [Monomial(-1, Y)] # -a + 5x + 6y == 5  ->  5x + 6y -5 == a
        return [new_var_constraint, Y]

    def convert_to_CNF(self, DNF_complements):
        """
        Converts complements of constraints in DNF to CNF
        using Max.
        Populates self.constraint_complements.
        """
        CNF_complements = []
        n_clauses = len(DNF_complements)
        if n_clauses == 0:
            pass

        elif n_clauses == 1:
            # define new var: Y = 5x + 6y - 5
            Y_definition, Y = self.cnf_conversion_helper(DNF_complements[0])
            # we want Y >= 0
            Y_ineq = Constraint(ConstraintType('GREATER_EQ'), monomials=[Monomial(1, Y)], scalar=0)
            CNF_complements.extend([Y_definition, Y_ineq])

        else: # nclauses > 1
            # handle first ineq
            Y_def, Y = self.cnf_conversion_helper(DNF_complements[0])
            CNF_complements.append(Y_def)
            # then disjunct all the complements using MaxConstraint
            for c in DNF_complements[1:]:
                # take complement, and turn into >= inequality
                Z_def, Z = self.cnf_conversion_helper(c)
                # begin disjunct train max(Z,Y) >= 0 ...
                Q = self.get_new_var()
                ###########################################################
                # changing max to be represented with Relu
                YmZ = self.get_new_var()
                YmZdef = Constraint('EQUALITY', monomials=[Monomial(1, Y), Monomial(-1, Z), Monomial(-1, YmZ)], scalar=0)
                RYmZ = self.get_new_var()
                RYmZdef = ReluConstraint(varin=YmZ, varout=RYmZ)
                # Q = relu(Y-Z) + Z
                max_constraint = Constraint('EQUALITY', monomials=[Monomial(1, RYmZ), Monomial(1, Z), Monomial(-1, Q)], scalar=0)
                ###########################################################
                # max_constraint = MaxConstraint((Y,Z), Q) # version with max
                # CNF_complements.extend([Z_def, max_constraint]) # version with max
                ############################################################    
                CNF_complements.extend([Z_def, YmZdef, RYmZdef, max_constraint])
                Y = Q
            # Q >= 0
            geq0 = Constraint(ConstraintType('GREATER_EQ'), monomials=[Monomial(1,Q)], scalar=0)
            CNF_complements.append(geq0)

        self.constraint_complements = CNF_complements
        return CNF_complements

        
