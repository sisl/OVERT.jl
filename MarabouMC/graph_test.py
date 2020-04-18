from termcolor import colored
import networkx as nx
import matplotlib.pyplot as plt
from keras.models import load_model
from MC_Keras_parser import KerasConstraint
from overt_to_python import OvertConstraint
from MC_constraints import Constraint, ConstraintType, MaxConstraint, ReluConstraint, MatrixConstraint, Monomial


class OvertConstraintGraph():
    def __init__(self, constraint_obj):
        self.constraints_obj = constraint_obj
        self.UnDiG = self.populate_graph(nx.Graph(), with_inequalities=True)
        self.DiG = self.populate_graph(nx.DiGraph(), with_inequalities=False)

    def populate_graph(self, G, with_inequalities=True):
        for eq in self.constraints_obj.constraints:
            if isinstance(eq, Constraint):
                if eq.type.type2str[eq.type._type] == "==":
                    self.populate_graph_eq(G, eq)
                elif with_inequalities:
                    self.populate_graph_ineq(G, eq)
            elif isinstance(eq, MatrixConstraint):
                self.popluate_graph_mateq(G, eq)
            elif isinstance(eq, MaxConstraint):
                self.populate_graph_maxeq(G, eq)
            elif isinstance(eq, ReluConstraint):
                self.populate_graph_relueq(G, eq)
            else:
                raise(IOError("This constraint is not recognized"))
        return G

    def populate_graph_eq(self, G, eq):
        vars = []
        coeffs = []
        for m in eq.monomials:
            vars.append(m.var)
            coeffs.append(m.coeff)


        if isinstance(self.constraints_obj, OvertConstraint):
            new_variable_coeff  = 1
        else:
            new_variable_coeff = -1

        if coeffs.count(1.0) == new_variable_coeff:
            idx = coeffs.index(1.0)
            tmp = vars[0]
            vars[0] = vars[idx]
            vars[idx] = tmp
        else:
            vars = sorted(vars, key=lambda x: float(x[2:]), reverse=True)

        for v in vars:
            if v not in G.nodes():
                G.add_node(v)
        for v in vars[1:]:
            G.add_edge(v, vars[0], color='r', weight=2, style="solid")

        return G

    def popluate_graph_mateq(self, G, eq):
        A = eq.A
        x = eq.x
        for i in range(A.shape[0]):
            new_eq = Constraint("EQUALITY", monomials=[])
            for j in range(A.shape[1]):
                Aij = A[i][j]
                if Aij != 0.:
                    new_eq.monomials.append(Monomial(coeff=Aij, var=x[j]))
            G = self.populate_graph_eq(G, new_eq)
        return G

    def populate_graph_maxeq(self, G, eq):
        if eq.var1in not in G.nodes(): G.add_node(eq.var1in)
        if eq.var2in not in G.nodes(): G.add_node(eq.var2in)
        if eq.varout not in G.nodes(): G.add_node(eq.varout)
        G.add_edge(eq.var1in, eq.varout, color='g', weight=2, style="solid")
        G.add_edge(eq.var2in, eq.varout, color='g', weight=2, style="solid")
        return G

    def populate_graph_relueq(self, G, eq):
        if eq.varin not in G.nodes(): G.add_node(eq.varin)
        if eq.varout not in G.nodes(): G.add_node(eq.varout)
        G.add_edge(eq.varin, eq.varout, color='b', weight=2, style="solid")
        return G

    def populate_graph_ineq(self, G, ineq):
        assert ineq.monomials[0].var in G.nodes()
        assert ineq.monomials[1].var in G.nodes()
        G.add_edge(ineq.monomials[0].var, ineq.monomials[1].var, color='y', weight=2, style="dashed")
        return G

    def plot_graph(self, G=None, figsize=(5, 5), pos_func=nx.spiral_layout, **kwargs):
        if G is None:
            G = self.DiG
        pos = pos_func(G)
        plt.figure(figsize=figsize)
        nx.draw_networkx_labels(G, pos, font_size=15)

        edges = G.edges()
        try:
            styles = [G[u][v]['style'] for u, v in edges]
            colors = [G[u][v]['color'] for u, v in edges]
            weights = [G[u][v]['weight'] for u, v in edges]
        except:
            colors = "red"
            weights = 2
            styles = "solid"

        nx.draw_networkx(G, pos, edges=edges, edge_color=colors, width=weights, style=styles, arrow=True, **kwargs)
        plt.show()

    def trim_graph(self, G, nodes):
        if G.is_directed():
            G_filtered = nx.DiGraph()
        else:
            G_filtered = nx.Graph()

        for e in G.edges:
            if (e[0] in nodes) or (e[1] in nodes):
                G_filtered.add_node(e[0])
                G_filtered.add_node(e[1])
                G_filtered.add_edge(e[0], e[1])
        return G_filtered

    def check_no_cycle(self):
        print("checking for cycles: ", end="")
        try:
            cyl = nx.find_cycle(self.DiG)
            print(colored("cycle found: ", "red"), cyl)
            return False
        except:
            print("no cycle found")
            return True

    def check_connectivity(self):
        if isinstance(self.constraints_obj, OvertConstraint):
            self._check_connectivity_overt()
        else:
            self._check_connectivity_keras()

    def _check_connectivity_keras(self):
        print("checking for connectivity: ", end="")
        if not nx.is_connected(self.UnDiG):
            print(colored("graph is not connected", "red"))
            return False
        else:
            print("graph is connected")
            print("checking for internal nodes connectivity: ", end="")
            for trg_node in self.UnDiG.nodes():
                test_failed = True
                input_nodes = [s for s in self.constraints_obj.model_input_vars if s in self.UnDiG.nodes]

                for src_node in input_nodes:
                    if trg_node in input_nodes:
                        test_failed = False
                        continue
                    if nx.has_path(self.UnDiG, source=src_node, target=trg_node):
                        test_failed = False
                        break
                if test_failed:
                    print("no input connection to node %s" % trg_node)
                    return False
            print("all internal nodes are connected to an input node")

            print("checking for internal nodes connectivity: ", end="")
            for src_node in self.UnDiG.nodes():
                test_failed = True
                output_nodes = [s for s in self.constraints_obj.model_output_vars if s in self.UnDiG.nodes]

                for trg_node in output_nodes:
                    if src_node in output_nodes:
                        test_failed = False
                        continue
                    if nx.has_path(self.UnDiG, source=src_node, target=trg_node):
                        test_failed = False
                        break
                if test_failed:
                    print("no input connection to node %s" % src_node)
                    return False

            print("all internal nodes are connected to an output node")
        print("connectivity passed.")
        return True

    def _check_connectivity_overt(self):

        print("checking for connectivity: ", end="")
        if not nx.is_connected(self.UnDiG):
            print("graph is not connected")
            return False
        else:
            print("graph is connected")
            print("checking for internal nodes connectivity: ", end="")
            for trg_node in self.UnDiG.nodes():
                test_failed = True
                input_nodes  = [s for s in self.constraints_obj.state_vars if s in self.UnDiG.nodes]
                input_nodes += [c for c in self.constraints_obj.control_vars if c in self.UnDiG.nodes]

                for src_node in input_nodes:
                    if trg_node in input_nodes:
                        test_failed = False
                        continue
                    if nx.has_path(self.UnDiG, source=src_node, target=trg_node):
                        test_failed = False
                        break
                if test_failed:
                    print("no input connection to node %s" % trg_node)
                    return False

            print("all internal nodes are connected to an input node")

            print("checking for internal nodes connectivity: ", end="")
            for src_node in self.UnDiG.nodes():
                output_node = self.constraints_obj.output_vars[0]
                if src_node in output_node:
                    continue

                if not nx.has_path(self.UnDiG, source=src_node, target=output_node):
                    print("no output connection from node %s" % src_node)
                    return False

            print("all internal nodes are connected to an output node")
        print("connectivity passed.")
        return True


if __name__ == "__main__":
    #overt_obj = OvertConstraint("../OverApprox/models/single_pend_acceleration_overt.h5")
    overt_obj = OvertConstraint("../OverApprox/models/double_pend_acceleration_1_overt.h5")
    graph = OvertConstraintGraph(overt_obj)
    print(graph.constraints_obj.constraints)
    graph.check_no_cycle()
    graph.check_connectivity()
    graph.plot_graph(G=graph.DiG, pos_func=nx.spring_layout, node_size=0, figsize=(10,10),
                     with_labels=False)  # nodelist=["xd1", "xd8", "xd10"]), edgelist,

    #model = load_model("/home/amaleki/Downloads/test_3_linear.h5")
    model = load_model("/home/amaleki/Downloads/test_55_linear.h5")
    keras_obj = KerasConstraint(model)

    graph = OvertConstraintGraph(keras_obj)
    print(graph.constraints_obj.constraints)
    graph.plot_graph(G=graph.DiG, pos_func=nx.spring_layout, node_size=0, figsize=(10, 10),
                     with_labels=False)

    graph.check_no_cycle()
    graph.check_connectivity()

