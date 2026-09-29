import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.datasets import QM9
from torch_geometric.utils import to_dense_batch, to_dense_adj
import time
from lib.layers import GraphConvolution, GraphAggregation, MultiDenseLayer



class Generator(nn.Module):
    def __init__(self, conv_dims, z_dim, vertexes, edges, nodes, dropout_rate):
        super(Generator, self).__init__()
        self.activation_f = torch.nn.Tanh()
        self.multi_dense_layer = MultiDenseLayer(z_dim, conv_dims, self.activation_f)
        self.vertexes = vertexes
        self.edges = edges
        self.nodes = nodes
        self.edges_layer = nn.Linear(conv_dims[-1], edges * vertexes * vertexes)
        self.nodes_layer = nn.Linear(conv_dims[-1], vertexes * nodes)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        output = self.multi_dense_layer(x)
        edges_logits = self.edges_layer(output).view(-1, self.edges, self.vertexes, self.vertexes)
        edges_logits = (edges_logits + edges_logits.permute(0, 1, 3, 2)) / 2
        edges_logits = self.dropout(edges_logits.permute(0, 2, 3, 1))
        
        nodes_logits = self.nodes_layer(output)
        nodes_logits = self.dropout(nodes_logits.view(-1, self.vertexes, self.nodes))
        return edges_logits, nodes_logits


class Discriminator(nn.Module):
    def __init__(self, conv_dim, m_dim, b_dim, dropout_rate=0.):
        super(Discriminator, self).__init__()
        self.activation_f = torch.nn.Tanh()
        graph_conv_dim, aux_dim, linear_dim = conv_dim
        self.gcn_layer = GraphConvolution(m_dim, graph_conv_dim, b_dim, False, 0, dropout_rate)
        self.agg_layer = GraphAggregation(graph_conv_dim[-1] + m_dim, aux_dim, self.activation_f, False, 0, dropout_rate)
        self.multi_dense_layer = MultiDenseLayer(aux_dim, linear_dim, self.activation_f, dropout_rate=dropout_rate)
        self.output_layer = nn.Linear(linear_dim[-1], 1, bias=False)

    def forward(self, adj, hidden, node):
        adj_conv = adj[:, :, :, 1:].permute(0, 3, 1, 2)
        h_1 = self.gcn_layer(node, adj_conv, hidden)
        h = self.agg_layer(h_1, node, hidden)
        h = self.multi_dense_layer(h)
        output = self.output_layer(h)
        return output


def gradient_penalty(D, real_nodes, real_adj, fake_nodes, fake_adj, device):
    alpha = torch.rand(real_nodes.size(0), 1, 1).to(device)
    alpha_adj = alpha.unsqueeze(-1) 
    
    int_nodes = (alpha * real_nodes + ((1 - alpha) * fake_nodes)).requires_grad_(True)
    int_adj = (alpha_adj * real_adj + ((1 - alpha_adj) * fake_adj)).requires_grad_(True)
    
    d_interpolates = D(int_adj, None, int_nodes)
    
    fake = torch.ones(real_nodes.shape[0], 1).to(device)
    gradients = autograd.grad(
        outputs=d_interpolates, inputs=(int_nodes, int_adj), grad_outputs=fake,
        create_graph=True, retain_graph=True, only_inputs=True
    )
    
    grad_nodes = gradients[0].view(gradients[0].size(0), -1)
    grad_adj = gradients[1].view(gradients[1].size(0), -1)
    grad_norm = torch.sqrt(torch.sum(grad_nodes ** 2, dim=1) + torch.sum(grad_adj ** 2, dim=1))
    return ((grad_norm - 1) ** 2).mean()
