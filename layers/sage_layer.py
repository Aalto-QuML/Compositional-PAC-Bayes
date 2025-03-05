from torch_geometric.nn import SAGEConv
from torch import nn

class SAGELayer(nn.Module):
    def __init__(
        self, in_features, out_features, activation, batch_norm, residual=False
    ):
        super().__init__()
        self.activation = activation
        self.batchnorm = nn.BatchNorm1d(out_features) if batch_norm else nn.Identity()

        self.residual = residual
        self.conv = SAGEConv(in_features, out_features)

    def forward(self, x, edge_index):
        h = self.conv(x, edge_index)
        h = self.batchnorm(h)
        h = self.activation(h)
        if self.residual:
            h = h + x
        return h