from visualizerV2 import Visualizer, Graph_interpreter
from optimizer import Optimizer
from PIL import Image
from tqdm import tqdm
import networkx as nx
import pandas as pd
import numpy as np
import imageio
import zipfile
import glob
import os
import io

class Tracer():
    def __init__(self,
                 associator,
                 stat_funcs,
                 node_trajectory,
                 max_occlusion,
                 quantile,
                 path,
                 dim = 2):

        self.dataset            = np.array((dataset := pd.read_csv('%s\\dataset.csv'%path)))
        self.columns            = dataset.columns
        index                   = pd.MultiIndex.from_tuples(list(map(tuple, np.array(self.dataset, dtype = np.uint16)[:,:2])))
        self.multi_indexed      = pd.DataFrame(self.dataset[:,:2].astype(np.uint16), index = index)
        self.path               = path

        self.stat_funcs         = stat_funcs
        self.optimizer          = Optimizer(self.stat_funcs)
        self.associator         = associator
        self.node_trajectory    = node_trajectory
        self.dim                = dim

        self.time_range = np.array([np.min(self.dataset[:,0]), np.max(self.dataset[:,0])], dtype = int)

        self._initialize_graph()
        self._window_sweep(max_occlusion, quantile)

    def _window_sweep(self, max_occlusion, quantile):
        iterr = iter(Fib(max_occlusion))
        for i, window_width in enumerate(iterr):
            self._eradicate_unlikely_connections(quantile)
            self._time_sweep(window_width)
            self._reevaluate_connections()

    def _time_sweep(self, window_width):
        for time in tqdm(range(self.time_range[0], self.time_range[1]), desc = 'Window width - %i'%window_width):
            group1, group2 = self._get_groups(time, time + window_width)
            if len(group1) == len(group2) == 0: continue

            ascs_4_opti, Ys, ascs = self.associator(group1, group2)
            ascs1, ascs2 = zip(*ascs)
            X, all_likelihoods, Likelihood = self.optimizer.optimize((ascs_4_opti, Ys, len(group1) + len(group2)))

            for parents, children, x, likelihood in zip(ascs1, ascs2, X, all_likelihoods):
                if not x: continue
                edges = []
                for parent in parents:
                    try: parent = group1[parent].nodes[-1]
                    except: p1, t1 = np.zeros(2), -1e48
                    else: p1, t1 = np.array(self.data[parent][2:4]), self.data[parent][0]
                    for child in children:
                        try: child = group2[child].nodes[0]
                        except: p2, t2 = np.zeros(2), 1e48
                        else: p2, t2 = np.array(self.data[child][2:4]), self.data[child][0]
                        v = np.dot(p1, p2) / (t2-t1)
                        edges.append((parent, child, {'likelihood': likelihood, 'velocity' : v}))

                if any([any([self.data[x[0]][0] == time     for x in edges]),
                        any([self.data[x[1]][0] == time + 1 for x in edges]),
                        likelihood > self.decision_boundary]):
                    self.graph.add_edges_from(edges)

    def get_total_log_likelihood(self):
        likelihoods = nx.get_edge_attributes(self.graph, 'likelihood')
        likelihoods = np.array(list(likelihoods.values()), dtype = np.longdouble )
        return -np.sum(np.log10(likelihoods))

    def _get_groups(self, start, stop):
        nodes1, nodes2 = [], []
        nodes1 = list(map(tuple, np.array(self.multi_indexed.loc[slice(start, stop - 1), :])))
        nodes2 = list(map(tuple, np.array(self.multi_indexed.loc[slice(start + 1, stop), :])))
        nodes1 = [x for x in nodes1 if list(self.graph._succ[x]) == []]
        nodes2 = [x for x in nodes2 if list(self.graph._pred[x]) == []]
        return(list(map(self._get_trajectory, nodes1)), list(map(self._get_trajectory, nodes2)))

    def _get_trajectory(self, node0):
        nodes = [node0]
        functions = [lambda x: list(y for y in self.graph._pred[x].keys() if type(y) is not str),
                     lambda x: list(y for y in self.graph._succ[x].keys() if type(y) is not str)]
        direction = int(len(functions[0](node0)) > 0) - int(len(functions[1](node0)) > 0)

        if direction:
            i = int(direction > 0)
            f1, f2 = functions[1 - i], functions[i]
            while True:
                nodes_o_i = f1(nodes[-1])
                if len(nodes_o_i) == 1 and len(f2(nodes_o_i[0])) == 1:
                    nodes.append(nodes_o_i[0])
                else: break

        nodes.sort(key = lambda x: x[0])
        return self.node_trajectory(self.graph.subgraph(set(nodes)))

    def _reevaluate_connections(self):
            interpretation = Graph_interpreter(self.graph.copy(), self.special_nodes, self.node_trajectory)
            interpretation.events()
            print('Trajectory count :' + str(len(interpretation.trajectories)))
            for stat_func in self.stat_funcs:
                if stat_func.conditions[0] == 1:
                    if stat_func.conditions[1] == 1:
                        for trajecotry in tqdm(interpretation.trajectories, desc = 'Reeeeee evaluating!'):
                            for i in range(len(trajecotry) - 1):
                                pieces, edge = trajecotry.split(i)
                                nx.set_edge_attributes(self.graph, {edge: stat_func([pieces[0]], [pieces[1]])}, 'likelihood')
                    elif stat_func.conditions[1] == 'n':
                        for event in interpretation.Events:
                            if len(event[1]) > 1:
                                inn  = interpretation.trajectories[event[0][0]]
                                outs = [interpretation.trajectories[j] for j in event[1]]
                                edges = [(inn.nodes[-1], j.nodes[0]) for j in outs]
                                likelihood = stat_func([inn], outs)
                                nx.set_edge_attributes(self.graph, {edge: likelihood for edge in edges}, 'likelihood')
                    else:
                        for event in interpretation.Events:
                            if type(event[1]) == str:
                                inn  = interpretation.trajectories[event[0][0]]
                                edge = (inn.nodes[-1], event[1][0])
                                likelihood = stat_func([inn], [])
                                nx.set_edge_attributes(self.graph, {edge: likelihood}, 'likelihood')
                elif stat_func.conditions[0] == 'n':
                    for event in interpretation.Events:
                        if len(event[0]) > 1:
                            ins  = [interpretation.trajectories[j] for j in event[0]]
                            out = interpretation.trajectories[event[1][0]]
                            edges = [(j.nodes[-1], out.nodes[0]) for j in ins]
                            likelihood = stat_func(ins, [out])
                            nx.set_edge_attributes(self.graph, {edge: likelihood for edge in edges}, 'likelihood')
                else:
                    for event in interpretation.Events:
                        if type(event[0]) == str:
                            out  = interpretation.trajectories[event[1][0]]
                            edge = (event[0][0], out.nodes[0])
                            likelihood = stat_func([], [out])
                            nx.set_edge_attributes(self.graph, {edge: likelihood}, 'likelihood')

    def _eradicate_unlikely_connections(self, quantile):
        likelihoods             = nx.get_edge_attributes(self.graph, 'likelihood')
        self.decision_boundary  = np.quantile(np.array(list(likelihoods.values())), quantile)
        removables              = [edge for edge in self.graph.edges if likelihoods[edge] <= self.decision_boundary]
        self.graph.remove_edges_from(removables)

    def _initialize_graph(self):
        self.graph         = nx.DiGraph()
        self.special_nodes = ['Entry', 'Exit']
        self.graph.add_nodes_from(self.special_nodes, data = 'mommy calls me speshal', position = 'special education class', params = 'speshal')

        for adress, point in zip(np.array(self.dataset)[:,:2], np.array(self.dataset)):
            node = tuple(map(int, adress))
            self.graph.add_node(node, data = list(point), position = point[2:2+self.dim], params = point[2 + self.dim:])

            for i, special_node in enumerate(self.special_nodes):
                edge = [special_node, node]
                if i: edge.reverse()
                self.graph.add_edge(*tuple(edge), likelihood = float(point[0] == self.time_range[i]), velocity = 1e-12)

        self.data       = nx.get_node_attributes(self.graph, 'data')
        self.position   = nx.get_node_attributes(self.graph, 'position')
        self.params     = nx.get_node_attributes(self.graph, 'params')

    def dump_data(self, sub_folder = None, images = None, memory = 15, smallest_trajectories = 1):
        if images is None: self.images = unzip_images('%s\\Compressed Data\\Shapes.zip'%self.path)
        else: self.images = images
        self.shape = self.images[0].shape
        total_likelihood = self.get_total_log_likelihood()

        if sub_folder is None:  output_path = self.path + '/Tracer Output'
        else:                   output_path = self.path + '/Tracer Output' + sub_folder + '_' + str(total_likelihood)
        try: os.makedirs(output_path)
        except: pass

        def stringizer(value):
            if type(value) == np.ndarray: return str(list(value))
            else: return str(value)
        nx.readwrite.gml.write_gml(self.graph, output_path + '/graph_%i.gml'%total_likelihood, stringizer = stringizer)
        interpretation = Graph_interpreter(self.graph, self.special_nodes, self.node_trajectory)
        interpretation.events()
        interpretation.families()
        Vis = Visualizer(self.images, interpretation)

        try: os.mkdir(output_path + '/trajectories')
        except FileExistsError:
            try: os.remove(output_path + '/trajectories/events.csv')
            except: pass

        #Trajectory visualization
        path = output_path + '/trajectories/Images'
        try: os.makedirs(path)
        except FileExistsError: map(os.remove, glob.glob(path + '/**.jpg'))
        Vis.ShowTrajectories(path)

        #History visualization
        path = output_path + '/tracedIDs'
        try: os.makedirs(path)
        except FileExistsError: map(os.remove, glob.glob(path + '/**.jpg'))
        Vis.ShowHistory(path, memory, smallest_trajectories, 'ID')

        #Family viusalization
        path = output_path + '/family_ID_photos'
        try: os.makedirs(path)
        except FileExistsError: map(os.remove, glob.glob(path + '/**.jpg'))
        Vis.ShowFamilies(path, 'ID')

        #Trajectory csv output
        try: os.makedirs(output_path + '/trajectories/changes')
        except FileExistsError: map(os.remove, glob.glob(output_path + '/trajectories/changes/**.csv'))
        try: os.makedirs(output_path + '/trajectories/data')
        except FileExistsError: map(os.remove, glob.glob(output_path + '/trajectories/data/**.csv'))

        cols = ['dt'] + ['d'+x for x in self.columns[2:]] + ['likelihoods']
        for i, track in enumerate(interpretation.trajectories):
            table = pd.DataFrame(data = track.data, columns = self.columns)
            table.to_csv(output_path + '/trajectories/data/data_%i.csv'%i, index = False)

            table = pd.DataFrame(data = track.changes, columns = cols)
            table.to_csv(output_path + '/trajectories/changes/changes_%i.csv'%i, index = False)

        with open(output_path + '/trajectories/events.csv', 'w') as file:
             events_str = 'Type,In,Out,Frame,X,Y,likelihood\n'
             for event in interpretation.Events:
                if type(event[0][0]) is str:
                    tr = interpretation.trajectories[event[1][0]].beginning
                    tmp_str = str([0] + event[0] + event[1] + [tr[0]] + list(tr[2:4]) + [event[2]])[1:-1] + '\n'
                elif type(event[1][0]) is str:
                    tr = interpretation.trajectories[event[0][0]].ending
                    tmp_str = str([1] + event[0] + event[1] + [tr[0]] + list(tr[2:4]) + [event[2]])[1:-1] + '\n'
                elif len(event[0]) > 1:
                    tr = interpretation.trajectories[event[1][0]].beginning
                    tmp_str = str([2] + [str(event[0]).replace(',', ';')[1:-1]] + event[1] + [tr[0]] + list(tr[2:4]) + [event[2]])[1:-1] + '\n'
                elif len(event[1]) > 1:
                    tr = interpretation.trajectories[event[0][0]].ending
                    tmp_str = str([3] + event[0] + [str(event[1]).replace(',', ';')[1:-1]] + [tr[0]] + list(tr[2:4]) + [event[2]])[1:-1] + '\n'
                events_str +=  tmp_str
                file.write(events_str)
            #Family graph output
        try: os.mkdir(output_path + '/family_graphs')
        except FileExistsError: map(os.remove, glob.glob(output_path + '/family_graphs/**.gml'))
        for i, family_graph in tqdm(enumerate(interpretation.families), desc = 'Writing family graphs: '):
            nx.readwrite.gml.write_gml(family_graph, output_path + '/family_graphs/family_%i.gml'%i, stringizer = str)
    
        del Vis, self.images

def unzip_images(path):
    imgFromZip  = lambda name: Image.open(io.BytesIO(zp.read(name)))
    with zipfile.ZipFile(path) as zp:
        names = zp.namelist()
        try:    names.remove(*[x for x in names if len(x.split('/')[-1]) == 0 or x.split(',')[-1] == 'ini'])
        except: pass
        names.sort(key = lambda x: int(x.split('/')[-1].split('_')[-1].split('.')[0]))
        images = list(map(imgFromZip, tqdm(names, desc = 'Loading images ')))

    scaler_f = lambda x: (2**-8 * int(np.max(x) >= 256) + int(np.max(x) < 256) + 254 * int(np.max(x) == 1))
    scaler   = scaler_f(np.asarray(images[15], np.uint16))

    mapper = lambda img:  np.repeat((np.asarray(img, np.uint16) * scaler).astype(np.uint8)[:,:,np.newaxis],3,2)
    images = list(map(mapper, tqdm(images, desc = 'Mapping images ')))

    return images

class Fib:
    def __init__(self, maxx):
        self.max = maxx

    def __iter__(self):
        self.a = 1
        self.b = 1
        return self

    def __next__(self):
        fib = self.a
        if fib > self.max: raise StopIteration
        self.a, self.b = self.b, self.a + self.b
        return fib
