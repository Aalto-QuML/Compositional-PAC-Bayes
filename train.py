import torch
from torch import linalg as LA
from bounds import regularizer as our_reg


def regularizer(model, alpha=1e-4):
    norms = []
    params = []
    for param in model.named_parameters():
        if 'theta' in param[0]:
            norms.append(LA.vector_norm(param[1].view(-1)) + 1.0)
            params.append(param[1].view(-1))
        if 'weight' in param[0] and 'transform_features' not in param[0]:
            params.append(param[1].view(-1))
            norms.append(LA.matrix_norm(param[1], ord=2))
    W = torch.stack(norms)
    beta = W.max()
    params = torch.concat(params, dim=0)
    aux1 = LA.vector_norm(params) ** 2
    l = torch.tensor(model.n_layers_final_mlp + 1).float().to(W.device)
    reg = alpha * torch.sqrt(l ** 2 * torch.log(l) * aux1 * (beta ** (2 * (l + 1))))
    return reg


def train(loader, model, loss_fn, optimizer, alpha, reg, device):
    model.train()
    train_losses = []
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = loss_fn(out.squeeze(), batch.y.squeeze()).mean()
        if reg == 'ours':
            r = our_reg(model, alpha=alpha)
            loss = loss + r
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    return train_losses


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    losses, accuracies = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        loss = loss_fn(out.squeeze(), batch.y.squeeze())
        loss = torch.sign(loss)
        acc = (out.argmax(dim=-1) == batch.y.squeeze()).float()
        losses.append(loss)
        accuracies.append(acc)
    signed_loss = torch.cat(losses, dim=0).view(-1).mean()
    acc = torch.cat(accuracies, dim=0).view(-1).mean()
    return signed_loss, acc


