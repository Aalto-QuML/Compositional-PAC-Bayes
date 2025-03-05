import torch
from torch import nn
from torch_geometric.nn import global_add_pool, global_mean_pool

from layers.gcn_factory import GcnCreator
from layers.gin_factory import GinCreator
from layers.sage_factory import SageCreator


class GNN(nn.Module):
    def __init__(
        self,
        gnn,
        hidden_dim,
        depth,
        num_node_features,
        global_pooling,
        batch_norm=True,
    ):
        super().__init__()
        if gnn == "gin":
            gnn_instance = GinCreator(hidden_dim, batch_norm)
        elif gnn == "gcn":
            gnn_instance = GcnCreator(hidden_dim, batch_norm)
        elif gnn == "sage":
            gnn_instance = SageCreator(hidden_dim, batch_norm)

        build_gnn_layer = gnn_instance.return_gnn_instance
        if global_pooling == "mean":
            graph_pooling_operation = global_mean_pool
        elif global_pooling == "sum":
            graph_pooling_operation = global_add_pool

        self.pooling_fun = graph_pooling_operation
        self.embedding = torch.nn.Linear(num_node_features, hidden_dim)

        layers = [build_gnn_layer(is_last=i == (depth - 1)) for i in range(depth)]

        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index, batch):
        x = self.embedding(x.float())

        for layer in self.layers:
            x = layer(x, edge_index=edge_index)

        x = self.pooling_fun(x, batch)
        return x