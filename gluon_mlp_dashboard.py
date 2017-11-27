from itertools import chain, product
import numpy as np
import mxnet as mx
from bqplot import *
from bqplot.marks import Graph
from ipywidgets import IntSlider, Dropdown, RadioButtons, HBox, VBox, Button, Layout
from bqplot import pyplot as plt
from bqplot import OrdinalScale

from IPython.display import display


class MLPDashboard(VBox):
    def __init__(self, net, path, name, **kwargs):
        self.net = net
        self.path = path
        self.name = name
        self.data = kwargs.pop('data', None)
        self.data = mx.nd.array(self.data)
        self.ctx = kwargs.pop('ctx', mx.cpu())
        self.num_epochs = kwargs.pop('num_epochs', 10)
        self.height = kwargs.get('height', 800)
        self.width = kwargs.get('width', 900)
        self.directed_links = kwargs.get('directed_links', False)

        self.get_shapes()
        self.layer_colors = kwargs.get('layer_colors',
                                       ['Orange'] * (len(self.num_hidden_layers) + 2))

        self.build_net()
        self.create_charts()
        self.graph.observe(self.hovered_change, 'hovered_point')

        super(MLPDashboard, self).__init__(children=[self.controls, self.figure], **kwargs)
        self.layout = layout=Layout(min_height='1000px')

    def load_epoch(self, epoch_num):
        file_path = self.path + '/' + self.name + '-' + str(epoch_num) +'.params'
        self.net.load_params(file_path, ctx=self.ctx)

    def get_weights_for_node_at_layer(self, epoch_num, layer_num, node_num):
        self.load_epoch(epoch_num)
        weights = self.net.collect_params()[list(self.net.collect_params())[2*(layer_num - 1)]].data().asnumpy()
        node_weights = weights[node_num-1, :]
        return node_weights

    def get_activations_hist(self, epoch, layer, node):
        if self.data is None:
            return
        self.load_epoch(epoch)
        outputs = self.net(self.data)
        self.graph.tooltip = self.hist_figure
        self.hist_figure.title = 'Activation Histogram for {}th Node at the {}th Layer - Epoch {}'.format(node, layer, epoch)
        self.hist_plot.sample = outputs[len(self.num_hidden_layers) - (layer - 1)][:, node].asnumpy()

    def update_bar_chart(self, layer, node):
        epoch = self.epoch_slider.value

        if self.mode_dd.value == 'Activations':
            self.get_activations_hist(epoch, layer, node)
            return

        if layer == 0:
            self.bar_plot.x = []
            self.bar_plot.y = []
            return

        if self.mode_dd.value == 'Weights':
            display_vals = self.get_weights_for_node_at_layer(epoch, layer, node)
        elif self.mode_dd.value == 'Gradients':
            display_vals = self.get_gradients_for_node_at_layer(epoch, layer, node)

        self.bar_figure.title = self.mode_dd.value + ' for layer:' + str(layer) + ' node: ' + str(node) + ' at epoch: ' + str(epoch)
        self.bar_plot.x = np.arange(len(display_vals))
        self.bar_plot.y = display_vals
        self.graph.tooltip = self.bar_figure

    def hovered_change(self, change):
        point_index = change['new']
        if point_index is None:
            return
        else:
            for i, n in enumerate(self.node_counts):
                if point_index < n:
                    break
                else:
                    point_index = point_index - n
            self.update_bar_chart(i, point_index)

    def get_gradients_for_node_at_layer(self, epoch_num, layer_num, node_num):
        self.load_epoch(epoch_num)
        grads_weights = self.net.collect_params()[list(self.net.collect_params())[2*(layer_num - 1)]].grad().asnumpy()
        node_gradients = grads_weights[node_num-1, :]
        return node_gradients

    def get_shapes(self):
        shapes = []
        for layer in list(self.net.collect_params()):
            if self.net.prefix + 'dense0_weight' in layer:
                self.num_inputs = self.net.collect_params()[layer].shape[1]
            if 'weight' in layer:
                shapes.append(self.net.collect_params()[layer].shape[0])
        self.num_hidden_layers = shapes[:-1]
        self.nodes_output_layer = shapes[-1]
        self.node_counts = [self.num_inputs] + self.num_hidden_layers + [self.nodes_output_layer]

    def create_charts(self):
        self.epoch_slider = IntSlider(description='Epoch:', min=1, max=self.num_epochs, value=1)
        self.mode_dd = Dropdown(description='View', options=['Weights', 'Gradients', 'Activations'], value='Weights')
        self.update_btn = Button(description='Update')

        self.bar_figure = plt.figure()
        self.bar_plot = plt.bar([], [], scales={'x': OrdinalScale()},
        colors=['Purple'])

        self.hist_figure = plt.figure(title='Histogram of Activations')
        self.hist_plot = plt.hist([], bins=20)

        self.controls = HBox([self.epoch_slider, self.mode_dd, self.update_btn])
        self.graph.tooltip = self.bar_figure

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
        self.link_matrix = np.empty((n,n))
        self.link_matrix[:] = np.nan

        for i in range(len(self.layer_nodes) - 1):
            curr_layer_nodes_indices = [node_indices[d] for d in self.layer_nodes[i]]
            next_layer_nodes = [node_indices[d] for d in self.layer_nodes[i+1]]
            for s, t in product(curr_layer_nodes_indices, next_layer_nodes):
                self.link_matrix[s, t] = 1

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

        self.graph = Graph(node_data=[{'label': d,
                                       'label_display': 'none'} for d in self.flattened_layer_nodes],
                           link_matrix=self.link_matrix, link_type='line',
                           colors=self.node_colors, directed=self.directed_links,
                           scales={'x': xs, 'y': ys}, x=self.nodes_x, y=self.nodes_y)
        self.graph.hovered_style = {'stroke': '1.5'}
        self.graph.unhovered_style = {'opacity': '0.1'}
        self.graph.selected_style = {'opacity': '1',
                                     'stroke': 'red',
                                     'stroke-width': '2.5'}
        self.figure = Figure()
        self.figure.marks = [self.graph]
        self.figure.title = 'Analyzing the Trained Neural Network'
        self.figure.layout.width = str(self.width) + 'px'
        self.figure.layout.height = str(self.height) + 'px'
