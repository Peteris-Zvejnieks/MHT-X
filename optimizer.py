from algorithm_x import AlgorithmX
import networkx as nx
import numpy as np

'''
Finds the combination of proposed associations which make sense and has the highest
likelihood given a list of statistical functions as objects of class statFunc.
'''

class Optimizer():
    class Likelihood0(Exception): pass
    class CondtionsNotFound(Exception): pass

    def __init__(self, stat_funcs):
        self.stat_funcs = stat_funcs

    def optimize(self, associations):
        self._prep(associations)
        X = np.zeros(self.num, dtype = np.uint8)

        for collection in self.collections:
            Universe, associations, likelihoods = (), [], np.zeros((2, len(collection)))
            for j, i in enumerate(collection):
                associations.append(self.flattened_associations[i])
                Universe += self.flattened_associations[i]
                likelihoods[:, j] = self.all_likelihoods[:, i]

            pos = np.arange(len(collection))
            Universe = tuple(set(Universe))
            mapped_associations = [tuple(map(lambda x: Universe.index(x), asc)) for asc in associations]

            search_space = optimization_reducer(len(Universe), mapped_associations)
            self.iter = search_space.generator()

            mem = [0, 0]
            for x in self.iter:
                likelihood = np.prod(likelihoods[(x, pos)])
                if likelihood > mem[1]: mem = [x, likelihood]
            if mem[1] == 0: raise Optimizer.Likelihood0(len(X))
            for j, i in enumerate(collection): X[i] = mem[0][j]
        return [X, self.all_likelihoods[(X, self.pos)], np.prod(self.all_likelihoods[(X, self.pos)])]

    def _prep(self, associations):
        self.associations, self.Ys, self.n = associations
        self.num = len(self.associations)
        self.pos = np.arange(self.num)
        self._calculate_likelihoods()
        self._find_disjoint_collections()

    def _find_disjoint_collections(self):
        graph = nx.Graph()
        graph.add_nodes_from(list(range(self.n)))

        for association in self.associations:
            for parent in association[0]:
                for child in association[1]:
                    graph.add_edge(parent, child)

        sub_graphs = (graph.subgraph(c) for c in nx.connected_components(graph))
        self.flattened_associations = list(map(lambda asc: asc[0] + asc[1], self.associations))
        self.collections = [[i for i, asc in enumerate(self.flattened_associations) if asc[0] in sub_graph] for sub_graph in sub_graphs]

    def _calculate_likelihoods(self):
        positive_likelihoods = np.zeros(self.pos.size, dtype = np.float64)
        for i, measurments in enumerate(self.Ys):
            positive_likelihoods[i] = self.main_likelihood_func(*measurments)
        positive_likelihoods = positive_likelihoods[:, np.newaxis].T
        negative_likelihoods = 1 - positive_likelihoods
        self.all_likelihoods = np.concatenate((negative_likelihoods, positive_likelihoods))

    '''
    Finds the apropriate statistical function for every association passed to it
    and applies it.
    If the likelihood is not in (0; 1], error is raised
    If no function is found for an association, Error is raised
    '''
    def main_likelihood_func(self, measurments1, measurments2):
        for stat_func in self.stat_funcs:
            if stat_func.check_conditions(measurments1, measurments2):
                likelihood = stat_func(measurments1, measurments2)
                if not 1 >= likelihood > 0: print('Warning, ' + str(stat_func.conditions)+ ' => %.4f'%likelihood)
                return likelihood
        optimizer.CondtionsNotFound(str(measurments1) + ';' + str(measurments2))

class optimization_reducer():
    def __init__(self, num, associations):
        self.num = len(associations)
        self.solver = AlgorithmX(num)
        for asc in associations: self.solver.appendRow(asc)

    def generator(self):
        for solution in self.solver.solve():
            X = np.zeros(self.num, dtype = np.uint8)
            X[solution] = 1
            yield X
