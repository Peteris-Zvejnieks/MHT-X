import  numpy as np
from matplotlib import cm
from tqdm import tqdm
import colorsys
import imageio
import copy
import cv2
import networkx as nx

class discrete_colormap():
    def __init__(self, N, /, hue = 1, sat = 0.8, val = 0.8):
        self.hue = lambda n: (hue *  n%N) /  N
        self.sat = lambda: sat + (1 - sat) * np.random.random()
        self.val = lambda: val + (1 - val) * np.random.random()
    def __call__(self, n): return colorsys.hsv_to_rgb(self.hue(n), self.sat(), self.val())

class Colorbar_overlay():
    def __init__(self, cmap, shape, /, relative_size = [0.5, 0.02], relative_pos = [0.3, 0.05]):
        relative_size, relative_pos, shapeXY  = np.array(relative_size), np.array(relative_pos), np.array(shape[:-1])
        self.cb_shape = (shapeXY * relative_size).astype(np.int)
        self.pos = (shapeXY * relative_pos).astype(np.int)

        gradient = np.swapaxes(cmap(np.linspace(1, 0, self.cb_shape[0]))[:,:-1,np.newaxis], 1, 2)
        colorbar = 255 * np.repeat(gradient, self.cb_shape[1], axis = 1)
        self.overlay = np.zeros(shape)
        self.overlay[self.pos[0]: self.pos[0] + self.cb_shape[0], self.pos[1]: self.pos[1] + self.cb_shape[1]] = colorbar

    def __call__(self, min_max):
        overlay = copy.deepcopy(self.overlay)
        overlay = cv2.putText(overlay, "{:.2f}".format(min_max[0]), (self.pos[1], self.pos[0] + self.cb_shape[0] + 30),    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
        overlay = cv2.putText(overlay, "{:.2f}".format(min_max[1]), (self.pos[1], self.pos[0] - 10),                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
        return overlay

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

class Visualizer():
    def __init__(self, images, interpretation, /, width = 2, upscale = 1):
        self.shape = tuple(map(lambda x: int(x*upscale), images[0].shape[:-1])) + (3,)
        self.upscale = upscale
        if upscale != 1: self.images = list(map(lambda img: cv2.resize(img, (self.shape[1],self.shape[0]), interpolation = cv2.INTER_CUBIC), images))
        else: self.images = images
        self.interpretation = interpretation
        self.trajectories = self.interpretation.trajectories
        self.special_nodes = interpretation.special_nodes
        self.width = width
        self.data = nx.get_node_attributes(self.interpretation.graph, 'data')

    def _get_node_crd(self, node):  return (int(self.data[node][2] * self.upscale), self.shape[0] - int(self.data[node][3] * self.upscale))
    def _map_color(self, color):    return tuple(map(lambda x: 255*x, color))[:3]
    def _normalizer(self, value, min_max):
        num = value - min_max[0]
        if (den := min_max[1] - min_max[0]) == 0:   return 1 - 1e-12
        else:                                       return min(num / den, 1 - 1e-12)

    def _draw_edge(self, img, edge, color):
        u, v = edge
        try: crd1 = self._get_node_crd(u)
        except: img = cv2.circle(img, self._get_node_crd(v), int(2 * self.width), color, self.width)
        else:
            try: crd2 = self._get_node_crd(v)
            except: img = cv2.circle(img, crd1, int(3 * self.width), color, self.width)
            else: img = cv2.line(img, crd1, crd2, color, self.width)
        finally: return img

    def ShowFamilies(self, path, key = 'likelihood'):
        cmap = cm.plasma
        color_bar_gen = Colorbar_overlay(cmap, self.shape)
        families = self.interpretation.families
        family_photos = np.zeros((len(families),) + self.shape, dtype = np.uint8)

        for i, family in tqdm(enumerate(families), desc = 'Drawing families '):
            family_photo = np.zeros(self.shape, dtype = np.uint8)
            if key != 'ID':
                edge_values = nx.get_edge_attributes(family, key)
                values = list(edge_values.values())
                for ID in family.nodes:
                    try: values.extend(list(nx.get_edge_attributes(self.trajectories[ID].backbone, key).values()))
                    except: continue
                min_max = [min(values), max(values)]
            else:
                dcmap = discrete_colormap(len(family.nodes))
                color_mapper = lambda indx: self._map_color(dcmap(indx))

            for j, ID in enumerate(family.nodes):
                try:
                    trajectory = self.trajectories[ID]
                    if key != 'ID':
                        values = nx.get_edge_attributes(trajectory.backbone, key)
                        color_mapper = lambda likelihood: self._map_color(cmap(self._normalizer(likelihood, min_max)))
                    else: color = color_mapper(j)
                    for edge in trajectory.backbone.edges:
                        if key != 'ID': color = color_mapper(values[edge])
                        family_photo = self._draw_edge(family_photo, edge, color)
                except TypeError: continue
                crd = self._get_node_crd(trajectory.nodes[int(len(trajectory.nodes)/2)])
                family_photo =  cv2.putText(family_photo, str(ID), crd, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)

            for edge in family.edges:
                if key != 'ID': color = color_mapper(edge_values[edge])
                if edge[0] == self.special_nodes[0]:
                    if key == 'ID': color = (0,  0, 255)
                    R = int(self.width * 2)
                    crd = self._get_node_crd(self.trajectories[edge[1]].nodes[0])
                    family_photo = cv2.circle(family_photo, crd, R, color, self.width)
                elif edge[1] == self.special_nodes[1]:
                    if key == 'ID': color = (0, 255, 0)
                    R = int(self.width * 3)
                    crd = self._get_node_crd(self.trajectories[edge[0]].nodes[-1])
                    family_photo = cv2.circle(family_photo, crd, R, color, self.width)
                else:
                    if key == 'ID':
                        if len(family._succ[edge[0]]) > 1:   color = (255, 0, 255)
                        elif len(family._pred[edge[1]]) > 1: color = (255, 0, 0)
                    edge = (self.trajectories[edge[0]].nodes[-1], self.trajectories[edge[1]].nodes[0])
                    family_photo = self._draw_edge(family_photo, edge, color)

            if key != 'ID':
                color_bar = color_bar_gen(min_max)
                family_photo = np.where(color_bar != 0, color_bar, family_photo)
            imageio.imwrite(path+'/%i.png'%i, family_photo)

    def ShowHistory(self, path, memory = 15, min_trajectory_size = 1, key = 'velocity'):
        events = self.interpretation.Events
        images = copy.deepcopy(self.images)
        dcmap = discrete_colormap(memory * 2)
        cmap = cm.plasma
        if key != 'ID': values = nx.get_edge_attributes(self.interpretation.graph, key)
        for i, tr in tqdm(enumerate(self.trajectories), desc = 'Drawing trajectories in ' + key + ' history ', total = len(self.trajectories)):
            if len(tr) < min_trajectory_size: continue
            if key != 'ID': min_max = [min(tuple(values.values()) + (1e3,)), max(tuple(values.values()) + (0,))]
            color = self._map_color(dcmap(i))
            for edge in zip(tr.nodes[:-1], tr.nodes[1:]):
                t0 = int(self.data[edge[1]][0])
                if key != 'ID': color = self._map_color(cmap(self._normalizer(values[edge], min_max)))
                for t in range(t0, min(len(images), t0 + memory)): images[t - 1] = self._draw_edge(images[t - 1], edge, color)
        for event in tqdm(events, desc = 'Drawing events in ' + key + ' history ', total = len(events)):
            stops, starts, likelihood = event
            if type(stops[0]) is str:  
                if len(self.trajectories[(ID := starts[0])]) < min_trajectory_size: continue
                t0 = int(self.trajectories[ID].data[0, 0])              
                f = lambda x: self._draw_entry(x, ID)
                
            elif type(starts[0]) is str:
                if len(self.trajectories[(ID := stops[0])]) < min_trajectory_size: continue
                t0 = int(self.trajectories[ID].data[-1, 0])
                f = lambda x: self._draw_exit(x, ID)
                
            elif len(stops) > 1:
                t0 = int(self.trajectories[starts[0]].data[0, 0])
                f = lambda x: self._draw_merger(x, stops, starts[0])
                
            elif len(starts) > 1:
                t0 = int(min([self.trajectories[x].data[0,0] for x in starts]))
                f = lambda x: self._draw_split(x, stops[0], starts)
                
            for t in range(t0, min(len(images), t0 + memory)): images[t - 1] = f(images[t - 1])
        for i, x in tqdm(enumerate(images), desc = 'Writing ' + key + ' history to disc ', total = len(images)): imageio.imwrite(path+'/%i.png'%i, x)

    def _draw_entry(self, img, ID):
        def square(img, crd, size, color, width):
            x, y = crd
            return cv2.rectangle(img, (x - size, y - size), (x + size, y + size), color,  width)
        
        crd = self._get_node_crd(self.trajectories[ID].nodes[0])
        return square(img, crd, 1, (0,  0, 255), self.width)   
        
    def _draw_exit(self, img, ID):
        def cross(img, crd, size, color, width):
            x, y = crd
            img = cv2.line(img, (x - size, y), (x + size, y), color,  width)
            return cv2.line(img, (x, y - size), (x, y + size), color, width)
        
        crd = self._get_node_crd(self.trajectories[ID].nodes[-1])
        return cross(img, crd, self.width, (0, 255, 0), int(self.width/2))


    def _draw_merger(self, img, starts, stop):
        color=  (255, 0, 0)
        crd2 = self._get_node_crd(self.trajectories[stop].nodes[0])
        for start in starts:
            crd1 = self._get_node_crd(self.trajectories[start].nodes[-1])
            img = cv2.arrowedLine(img, crd1, crd2, color, 1)
        return img

    def _draw_split(self, img, start, stops):
        color = (255, 0, 255)
        crd1 = self._get_node_crd(self.trajectories[start].nodes[-1])
        for stop in stops:
            crd2 = self._get_node_crd(self.trajectories[stop].nodes[0])
            img = cv2.arrowedLine(img, crd1, crd2, color, self.width)
        return img

    def ShowTrajectories(self, path, key = 'velocity'):
        cmap = cm.plasma
        color_bar_gen = Colorbar_overlay(cmap, self.shape)
        for i, tr in tqdm(enumerate(self.trajectories), desc = 'Drawing trajectories '):
            if len(tr) == 1: continue
            img = np.zeros(self.shape, dtype=np.uint8)
            values = nx.get_edge_attributes(tr.backbone, key)
            min_max = [min(tuple(values.values()) + (1e3,)), max(tuple(values.values()) + (0,))]
            for edge in zip(tr.nodes[:-1], tr.nodes[1:]):
                color = self._map_color(cmap(self._normalizer(values[edge], min_max)))
                img = self._draw_edge(img, edge, color)
            color_bar = color_bar_gen(min_max)
            img = np.where(color_bar != 0, color_bar, img)
            img = img.astype(np.uint8)
            imageio.imwrite(path+'/%i.png'%i, img)
