import argparse
from torch import tensor
import numpy as np
import os
import json
from train import *

from torch_geometric.loader import DataLoader

from data.datasets import get_data
from perslay import PersLay
import torch.optim as optim
import torch.nn as nn
import copy

parser = argparse.ArgumentParser(description='PersLay!')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--logdir', type=str, default='results', help='Log directory')
parser.add_argument('--dataset', type=str, default='NCI1',
                    choices=['DHFR', 'MUTAG', 'COX2', 'PROTEINS', 'NCI109', 'NCI1', 'IMDB-BINARY', 'IMDB-MULTI', 'ogbg-molhiv'])
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
parser.add_argument('--max_epochs', type=int, default=2000, help='Number of epochs to train.')
parser.add_argument('--interval', type=int, default=1, help='Interval for printing train statistics.')
parser.add_argument('--width_weight_fn', type=int, default=64)
parser.add_argument('--width_final_mlp', type=int, default=64)
parser.add_argument('--n_layers_weight_fn', type=int, default=1)
parser.add_argument('--n_layers_final_mlp', type=int, default=2)
parser.add_argument('--q', type=int, default=64)
parser.add_argument('--gnn_hidden', type=int, default=64)
parser.add_argument('--gnn_depth', type=int, default=2)
parser.add_argument("--no-bn", dest="bn", action="store_false")

parser.add_argument('--point_transform', type=str, default='gaussian', choices=['gaussian', 'line', 'triangle'])
parser.add_argument('--agg_type', type=str, default='mean', choices=['mean', 'sum', 'max'])
parser.add_argument('--gamma', type=float, default=1.0)
parser.add_argument('--gnn', type=str, default='gcn', choices=['gin', 'gcn', 'sage'])
parser.add_argument('--alpha', type=float, default=1e-6)  # regularization factor
parser.add_argument("--regularizer", choices=["none", "ours", "wd"], default="ours")
parser.add_argument("--no-gnn", dest="use_gnn", action="store_false")
parser.add_argument('--use_weight_fn',  action="store_true")

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

args.logdir = f'{args.logdir}/{args.dataset}/{args.point_transform}/{args.agg_type}/' \
              f'{args.n_layers_weight_fn}/{args.width_weight_fn}/{args.n_layers_final_mlp}/{args.width_final_mlp}/' \
              f'{args.q}/use_weight_{args.use_weight_fn}/{args.regularizer}/alpha-{args.alpha}'

print(args.logdir)

if not os.path.exists(f'{args.logdir}'):
    os.makedirs(f'{args.logdir}')

if not os.path.exists(f'{args.logdir}/models/'):
    os.makedirs(f'{args.logdir}/models/')

with open(f'{args.logdir}/summary.json', 'w') as f:
    json.dump(args.__dict__, f, indent=2)

dataset, train_data, val_data, test_data = get_data(args.dataset)

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=False)
test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)

# Model
model = PersLay(width_weight_fn=args.width_weight_fn, width_final_mlp=args.width_final_mlp,
                n_layers_weight_fn=args.n_layers_weight_fn, n_layers_final_mlp=args.n_layers_final_mlp,
                num_classes=dataset.num_classes,
                q=args.q, point_transform=args.point_transform, agg_type=args.agg_type,
                use_gnn=args.use_gnn, gnn=args.gnn, gnn_hidden=args.gnn_hidden, gnn_depth=args.gnn_depth, bn=args.bn,
                num_graph_features=train_data[0].graph_features.shape[1], num_features=dataset.num_node_features, use_weight=args.use_weight_fn)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.alpha*int(args.regularizer == 'wd'))
loss_fn = nn.MultiMarginLoss(reduction='none')

train_losses = []
train_signed_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []
val_losses = []
val_accuracies = []

models = []

for epoch in range(1, args.max_epochs + 1):

    train_loss = train(train_loader, model, loss_fn, optimizer, args.alpha, args.regularizer, device)

    train_signed_loss, train_acc = evaluate(model, train_loader, loss_fn, device)
    val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
    test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)

    train_accuracies.append(train_acc)
    train_signed_losses.append(train_signed_loss)

    test_accuracies.append(test_acc)
    test_losses.append(test_loss)

    val_accuracies.append(val_acc)
    val_losses.append(val_loss)

    train_losses.append(torch.tensor(train_loss).mean())

    if (epoch - 1) % args.interval == 0:
        print(
            f"{epoch:3d}: Train Loss: {torch.tensor(train_loss).mean():.3f},"
            f" Val Loss: {val_loss:.3f}, Val Acc: {val_accuracies[-1]:.3f}, "
            f"Test Loss: {test_loss:.3f}, Test Acc: {test_accuracies[-1]:.3f}"
        )

    if epoch in [1, 10, 20, 30]:
        models.append(copy.deepcopy(model.state_dict()))

#torch.save(models, f'{args.logdir}/models/perslay_{args.seed}.models')

results = {
    "train_losses": tensor(train_losses),
    "train_signed_losses": tensor(train_signed_losses),
    "train_accuracies": tensor(train_accuracies),
    "test_accuracies": tensor(test_accuracies),
    "test_losses": tensor(test_losses),
    "val_accuracies": tensor(val_accuracies),
    "val_losses": tensor(val_losses),
}

#torch.save(results, f'{args.logdir}/perslay_{args.seed}.results')
