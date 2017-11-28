from itertools import chain, product
import numpy as np
from bqplot import *
from bqplot.marks import Graph
from ipywidgets import IntSlider, Dropdown, RadioButtons, HBox, VBox, Button, Layout
from bqplot import pyplot as plt
from bqplot import OrdinalScale, OrdinalColorScale
from bqplot.colorschemes import CATEGORY10

from IPython.display import display


class NeuralNet(Figure):
    def __init__(self, **kwargs):
        self.height = kwargs.get('height', 800)
        self.width = kwargs.get('width', 900)
        self.directed_links = kwargs.get('directed_links', False)

        self.num_inputs = kwargs['num_inputs']
        self.num_hidden_layers = kwargs['num_hidden_layers']
        self.nodes_output_layer = kwargs['num_outputs']
        self.layer_colors = kwargs.get('layer_colors',
                                       ['Orange'] * (len(self.num_hidden_layers) + 2))

        self.build_net()
        super(NeuralNet, self).__init__(**kwargs)

    def build_net(self):
        # create nodes
        self.layer_nodes = []
        self.layer_nodes.append(['x' + str(i+1) for i in range(self.num_inputs)])

        for i, h in enumerate(self.num_hidden_layers):
            self.layer_nodes.append(['h' + str(i+1) + ',' + str(j+1) for j in range(h)])
        self.layer_nodes.append(['y' + str(i+1) for i in range(self.nodes_output_layer)])

        self.flattened_layer_nodes = list(chain(*self.layer_nodes))

        # build link matrix
        i = 0
        node_indices = {}
        for layer in self.layer_nodes:
            for node in layer:
                node_indices[node] = i
                i += 1

        n = len(self.flattened_layer_nodes)
        self.link_data = []
        for i in range(len(self.layer_nodes) - 1):
            curr_layer_nodes_indices = [node_indices[d] for d in self.layer_nodes[i]]
            next_layer_nodes = [node_indices[d] for d in self.layer_nodes[i+1]]
            for s, t in product(curr_layer_nodes_indices, next_layer_nodes):
                self.link_data.append({'source': s, 'target': t, 'value': 0})

        # set node x locations
        self.nodes_x = np.repeat(np.linspace(0, 100,
                                             len(self.layer_nodes) + 1,
                                             endpoint=False)[1:],
                                 [len(n) for n in self.layer_nodes])

        # set node y locations
        self.nodes_y = np.array([])
        for layer in self.layer_nodes:
            n = len(layer)
            ys = np.linspace(0, 100, n+1, endpoint=False)[1:]
            self.nodes_y = np.append(self.nodes_y, ys[::-1])

        # set node colors
        n_layers = len(self.layer_nodes)
        self.node_colors = np.repeat(np.array(self.layer_colors[:n_layers]),
                                     [len(layer) for layer in self.layer_nodes]).tolist()

        xs = LinearScale(min=0, max=100)
        ys = LinearScale(min=0, max=100)
        link_color_scale = OrdinalColorScale(colors=['gray'] + CATEGORY10, domain=list(range(11)))

        self.graph = Graph(node_data=[{'label': d,
                                       'label_display': 'none'} for d in self.flattened_layer_nodes],
                           # link_matrix=self.link_matrix, 
                           link_data=self.link_data,
                           link_type='line',
                           colors=self.node_colors, directed=self.directed_links,
                           scales={'x': xs, 'y': ys, 'link_color': link_color_scale}, x=self.nodes_x, y=self.nodes_y)
        self.graph.hovered_style = {'stroke': '1.5'}
        self.graph.unhovered_style = {'opacity': '0.1'}
        self.graph.selected_style = {'opacity': '1',
                                     'stroke': 'red',
                                     'stroke-width': '2.5'}
        self.marks = [self.graph]
        self.title = 'Analyzing the Trained Neural Network'
        self.layout.width = str(self.width) + 'px'
        self.layout.height = str(self.height) + 'px'
