from layers.sage_layer import SAGELayer
from layers.gnn_factory_interface import GNNFactoryInterface
from torch import nn
import torch.nn.functional as F


class SageCreator(GNNFactoryInterface):
    def __init__(self, hidden_dim, batch_norm):
        self.hidden_dim = hidden_dim
        self.batch_norm = batch_norm

    def return_gnn_instance(self, is_last=False):
        return SAGELayer(
            self.hidden_dim,
            # num_node_features if is_first else hidden_dim,
            self.hidden_dim,
            # num_classes if is_last else hidden_dim,
            nn.Identity() if is_last else F.relu,
            batch_norm=self.batch_norm
        )