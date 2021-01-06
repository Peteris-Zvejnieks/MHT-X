import networkx as nx

class Graph_interpreter():
    def __init__(self, graph, special_nodes, node_trajectory):
        self.graph = graph
        self.special_nodes = special_nodes
        self.node_trajectory = node_trajectory
        self._trajectories()

    def _find_by_node(self, node):
        for i, x in enumerate(self.trajectories):
            if node == x.nodes[0] or node == x.nodes[-1]: return i

    def _trajectories(self):
        tmp_graph = self.graph.copy()
        tmp_graph.remove_nodes_from(self.special_nodes)
        for node in tmp_graph.nodes:
            if len(ins  := list(tmp_graph.in_edges( node))) > 1: tmp_graph.remove_edges_from(ins )
            if len(outs := list(tmp_graph.out_edges(node))) > 1: tmp_graph.remove_edges_from(outs)
        tmp_graph = tmp_graph.to_undirected()
        self.paths = list(self.graph.subgraph(c) for c in nx.connected_components(tmp_graph))
        self.paths.sort(key = lambda x: -len(x.nodes))
        self.trajectories = list(map(self.node_trajectory, self.paths))

    def events(self):
        self.Events = []
        for i, trajectory in enumerate(self.trajectories):
            ins  = self.graph._pred[trajectory.nodes[0]]
            if list(ins.keys())[0] == self.special_nodes[0]: self.Events.append([[self.special_nodes[0]], [i], list(ins.values())[0]['likelihood']])
            elif len(ins) > 1: self.Events.append([[self._find_by_node(x) for x in ins.keys()] , [i], list(ins.values())[0]['likelihood']])
            outs = self.graph._succ[trajectory.nodes[-1]]
            if list(outs.keys())[0] == self.special_nodes[1]: self.Events.append([[i], [self.special_nodes[1]], list(outs.values())[0]['likelihood']])
            elif len(outs) > 1: self.Events.append([[i], [self._find_by_node(x) for x in outs.keys()], list(outs.values())[0]['likelihood']])

    def families(self):
        smol_graph = nx.DiGraph()
        for event in self.Events:
            for source in event[0]:
                for sink in event[1]:
                    smol_graph.add_edge(source, sink, likelihood = event[2])

        tmp_graph = smol_graph.copy().to_undirected()
        tmp_graph.remove_nodes_from(self.special_nodes)
        self.families = list(smol_graph.subgraph(c.union(set(self.special_nodes))) for c in nx.connected_components(tmp_graph))
        self.families.sort(key = lambda graph: -len(list(graph.nodes)))
