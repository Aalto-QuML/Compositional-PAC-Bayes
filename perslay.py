import torch.nn as nn
import torch

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from gnn import GNN


class PersLay(nn.Module):
    def __init__(self, width_weight_fn, width_final_mlp,
                n_layers_weight_fn, n_layers_final_mlp,
                q, point_transform, agg_type, num_classes,
                 use_gnn, gnn, gnn_hidden, gnn_depth, bn,
                 num_graph_features, num_features, sigma=0.1, use_weight=False):
        super(PersLay, self).__init__()
        self.dim = q
        self.num_classes = num_classes
        self.point_transform = point_transform
        self.width_weight_fn = width_weight_fn
        self.width_final_mlp = width_final_mlp
        self.n_layers_weight_fn = n_layers_weight_fn
        self.n_layers_final_mlp = n_layers_final_mlp
        self.agg_type = agg_type

        if self.agg_type == 'max':
            self.pooling = global_max_pool
        elif self.agg_type == 'mean':
            self.pooling = global_mean_pool
        elif self.agg_type == 'sum':
            self.pooling = global_add_pool

        if point_transform == 'triangle':
            self.phi = self.triangle
            self.theta = torch.nn.Parameter(torch.randn(self.dim))

        if point_transform == 'gaussian':
            self.phi = self.gaussian
            self.theta = torch.nn.Parameter(torch.randn(self.dim, 2))

        if point_transform == 'line':
            self.phi = self.line
            self.theta = torch.nn.Parameter(torch.randn(self.dim, 3))

        layers = []
        for i in range(n_layers_final_mlp):
            layers.append(nn.Linear(in_features=width_final_mlp if i != 0 else  2*self.theta.shape[0] + num_graph_features + (gnn_hidden if use_gnn else 0),
                                    out_features=width_final_mlp))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(in_features=width_final_mlp if n_layers_final_mlp != 0 else 2*self.theta.shape[0] + num_graph_features + (gnn_hidden if use_gnn else 0),
                                out_features=num_classes))
        self.rho = nn.Sequential(*layers)

        self.sigma = sigma
        self.use_weight = use_weight

#        self.transform_features = torch.nn.Sequential(
#            nn.Linear(num_features, hidden // 2),
#            nn.ReLU(),
#            nn.Linear(hidden // 2, hidden // 4),
#            nn.ReLU(),
#            nn.Linear(hidden // 4, hidden // 4)
#        )
        self.use_gnn = use_gnn
        if use_gnn:
            self.gnn = GNN(gnn=gnn, hidden_dim=gnn_hidden, depth=gnn_depth,
                           num_node_features=num_features, global_pooling='sum', batch_norm=bn)
        if use_weight:
            weight_layers = []
            for i in range(n_layers_weight_fn):
                weight_layers.append(nn.Linear(in_features=width_weight_fn if i != 0 else 2, out_features=width_weight_fn))
                weight_layers.append(nn.ReLU())
            weight_layers.append(nn.Linear(in_features=width_weight_fn if n_layers_weight_fn != 0 else 2, out_features=1))
            weight_layers.append(nn.Sigmoid())
            self.weight = nn.Sequential(*weight_layers)

    def triangle(self, diags):
        aux1 = diags[:, :, 1].unsqueeze(-1).expand(diags.shape[0], diags.shape[1], self.theta.shape[0])
        aux2 = diags[:, :, 0].unsqueeze(-1).expand(diags.shape[0], diags.shape[1], self.theta.shape[0])
        aux3 = self.theta.expand_as(aux1)
        zeros = torch.zeros_like(aux3)
        output = torch.max(zeros, aux1 - torch.abs(aux3 - aux2))
        return output

    def gaussian(self, diags):
        return torch.exp(-torch.cdist(diags, self.theta)**2 / (2* self.sigma**2))

    def line(self, diags):
        A = torch.cat((diags[:, :, :2], torch.ones(diags.shape[0], diags.shape[1], 1).to(diags.device)), axis=-1)
        B = self.theta.unsqueeze(0).expand(diags.shape[0], -1, -1)
        C = A @ torch.transpose(B, 2, 1)
        return C

    def forward(self, inputs):

        if self.use_gnn:
            gnn_embedding = self.gnn(inputs.x, inputs.edge_index, inputs.batch)

        vectorized_diagrams = []
        for t in inputs.diagms.keys():
            representations = self.phi(inputs.diagms[t])
            if self.use_weight:
                x = self.weight(inputs.diagms[t])
                representations = x * representations
            ###
            slices = inputs._slice_dict.get("diagms")[t]
            diff_edge_slices = slices[1:] - slices[:-1]
            n_batch = len(diff_edge_slices)
            batch = torch.repeat_interleave(
                torch.arange(n_batch, device=inputs.x.device), diff_edge_slices
            )
            x = self.pooling(representations, batch)
            ###
            vectorized_diagrams.append(x)
        topological_embedding = torch.cat(vectorized_diagrams, dim=1).to(inputs.x.device)

        if self.use_gnn:
            concat_representations = torch.cat([topological_embedding, gnn_embedding, inputs.graph_features], dim=1)
        else:
            concat_representations = torch.cat([topological_embedding, inputs.graph_features], dim=1)

        final_representations = self.rho(concat_representations) if self.rho != "identity" else concat_representations
        return final_representations
