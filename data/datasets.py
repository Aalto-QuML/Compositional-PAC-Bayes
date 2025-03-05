import os.path as osp

import torch
from ogb.graphproppred import PygGraphPropPredDataset
from sklearn.model_selection import StratifiedShuffleSplit
from torch_geometric.datasets import ZINC, TUDataset
from torch_geometric.utils import get_laplacian
import torch_geometric, scipy
from gudhi.simplex_tree import SimplexTree
import itertools
import numpy as np
from scipy.sparse import csgraph
from scipy.linalg import eigh


class FilterConstant(object):
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, data):
        data.x = torch.ones(data.num_nodes, self.dim)
        return data


def get_ogb_data(name):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', name)
    dataset = PygGraphPropPredDataset(name=name, root=path)
    return dataset


def get_tu_datasets(name, feat_replacement='constant'):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', name)
    dataset = TUDataset(name=name, root=path)
    if not hasattr(dataset, 'x'):
        if feat_replacement == 'constant':
            dataset.transform = FilterConstant(10)
    return dataset

def get_data(name):
    if name == 'ogbg-molhiv':
        dataset = get_ogb_data(name)
        dataset = merge_diagrams(dataset)
        split_idx = dataset.get_idx_split()
        train_data = dataset[split_idx["train"]]
        val_data = dataset[split_idx["valid"]]
        test_data = dataset[split_idx["test"]]
    else:
        dataset = get_tu_datasets(name)
        dataset = merge_diagrams(dataset)
        train_data, val_data, test_data = data_split(dataset)
    return dataset, train_data, val_data, test_data


def data_split(dataset, seed=42):
    skf_train = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, val_test_idx = list(skf_train.split(torch.zeros(len(dataset)), dataset.y))[0]
    skf_val = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_idx, test_idx = list(skf_val.split(torch.zeros(val_test_idx.size), dataset.y[val_test_idx]))[0]
    train_data = dataset[train_idx]
    val_data = dataset[val_test_idx[val_idx]]
    test_data = dataset[val_test_idx[test_idx]]
    return train_data, val_data, test_data


def hks_signature(eigenvectors, eigenvals, time):
    return np.square(eigenvectors).dot(np.diag(np.exp(-time * eigenvals))).sum(axis=1)


def merge_diagrams(dataset, param_range=[0.1, 10]):
    data_list = []

    pad_size = 1
    for data in dataset:
        pad_size = np.max((data.num_nodes, pad_size))  # maximum number of nodes in the dataset

    for data in dataset:
#        L = get_laplacian(data.edge_index, normalization='sym')
#        L_dense = torch_geometric.utils.to_dense_adj(L[0], edge_attr=L[1], max_num_nodes=data.num_nodes)
#        values, vectors = torch.linalg.eigh(L_dense)
        num_vertices = data.num_nodes
        A = torch_geometric.utils.to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes).squeeze().numpy()

        L = csgraph.laplacian(A, normed=True)
        egvals, egvectors = eigh(L)
        eigenvectors = np.zeros([num_vertices, pad_size])
        eigenvals = np.zeros(pad_size)
        eigenvals[:min(pad_size, num_vertices)] = np.flipud(egvals)[:min(pad_size, num_vertices)]
        eigenvectors[:, :min(pad_size, num_vertices)] = np.fliplr(egvectors)[:, :min(pad_size, num_vertices)]

        multihead_dgms = {}
        graph_features = []
        graph_features.append(eigenvals)

        (xs, ys) = np.where(np.triu(A))

        for t in param_range:
            st = SimplexTree()

            filtration_values = hks_signature(egvectors, egvals, time=t)
#            filtration_values = (torch.exp(-t * values) * (vectors ** 2)).sum(dim=1).squeeze()
#            correct_idx = data.edge_index[0] < data.edge_index[1]
#            e_idx = data.edge_index[:, correct_idx]
#            for v in range(data.num_nodes):
#                st.insert([v], filtration=filtration_values[v])
#            for u, v in zip(e_idx[0, :], e_idx[1, :]):
#                st.insert(sorted([u, v]), max(filtration_values[v], filtration_values[u]))
#            st.extend_filtration()
#            dgms = st.extended_persistence()

            for i in range(num_vertices):
                st.insert([i], filtration=-1e10)
            for idx, x in enumerate(xs):
                st.insert([x, ys[idx]], filtration=-1e10)
            for i in range(num_vertices):
                st.assign_filtration([i], filtration_values[i])

            st.make_filtration_non_decreasing()
            st.extend_filtration()
            dgms = st.extended_persistence()
            joint_diagrams = list(itertools.chain.from_iterable(dgms))
            dgms_tensor = torch.tensor([pair for i, pair in joint_diagrams]).view(-1, 2)
            multihead_dgms[t] = dgms_tensor
            graph_features.append(
                np.percentile(hks_signature(eigenvectors, eigenvals, time=t), 10 * np.arange(11))
            )
        data.graph_features = torch.from_numpy(np.concatenate(graph_features)).unsqueeze(0).float()
        data.diagms = multihead_dgms

        data_list.append(data)
    dataset.data, dataset.slices = dataset.collate(data_list)
    return dataset
