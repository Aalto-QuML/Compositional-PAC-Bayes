from dataclasses import dataclass
import torch
from typing import Sequence

@dataclass
class Lemma2Variables:
    def __init__(self, T: torch.Tensor, S: torch.Tensor, C_norm: float, C_pert: float, eta: float, h: int, weight_matrices: Sequence[torch.Tensor]):
        self.T = T
        self.S = S
        self.C_norm = C_norm
        self.C_pert = C_pert
        self.eta = eta
        self.h = h
        self.weight_matrices = weight_matrices
        self.n = None
        self.__post_init__()

    def __post_init__(self):
        assert len(self.S.shape) == 1
        self.n = self.S.shape[0]
        assert self.eta <= self.C_norm / 2 / self.C_pert
#        print((1 / self.S).mean(), (1/self.T) ** (1 / self.n))
        assert (1 / self.S).mean() >= (1/self.T) ** (1 / self.n) - 10e-4


def from_gnn(torch_weight_matrices, prod_of_lipschitz=1., B=1., d=10):
    h = max(max(matrix.shape) for matrix in torch_weight_matrices)
    n = len(torch_weight_matrices)
    operator_norms = torch.stack([torch.svd(matrix, compute_uv=False).S[0] for matrix in torch_weight_matrices])

    d_power = d ** ((n - 1) / 2)
    C_norm = B * d_power * prod_of_lipschitz
    C_pert = torch.exp(torch.tensor(1.0)) * d_power * B * prod_of_lipschitz

    return Lemma2Variables(
        T=torch.prod(operator_norms),
        S=operator_norms,
        eta=1 / 6 / len(torch_weight_matrices),
        C_norm=C_norm,
        C_pert=C_pert,
        h=h,
        weight_matrices=torch_weight_matrices,
    )


def from_mlp(torch_weight_matrices, prod_of_lipschitz=1., B=1.):
    h = max(max(matrix.shape) for matrix in torch_weight_matrices)
    operator_norms = torch.stack([torch.svd(matrix, compute_uv=False).S[0] for matrix in torch_weight_matrices])

    C_norm = B * prod_of_lipschitz
    C_pert = torch.exp(torch.tensor(1.0)) * B * prod_of_lipschitz

    return Lemma2Variables(
        T=torch.prod(operator_norms),
        S=operator_norms,
        eta=1 / 6 / len(torch_weight_matrices),
        C_norm=C_norm,
        C_pert=C_pert,
        h=h,
        weight_matrices=torch_weight_matrices,
    )


def from_gaussian_perslay(point_transformation_function_weights, q=None, tau=1.0, A_1=1., A_2=1.,
                          weight_function_weights=None, b=1., Lip_weight=1., C_weight=1.):
    if q is None:
        q = point_transformation_function_weights.shape[0]

    T_varphi = torch.max(torch.tensor(1.0).to(point_transformation_function_weights.device), torch.norm(point_transformation_function_weights))
    T_weight = torch.max(torch.tensor(1.0).to(point_transformation_function_weights.device), torch.norm(weight_function_weights) if weight_function_weights is not None else torch.tensor(0).to(point_transformation_function_weights.device))

    # Calculate constants
    C_varphi = torch.sqrt(torch.tensor(1.0 * q, dtype=torch.float32))
    Lip_varphi = tau * torch.exp(torch.tensor(-0.5))

    # Calculate C_norm and C_pert
    C_norm = 2 * A_1 * C_weight * C_varphi * max(1,
                                                (A_2 * max(Lip_varphi, Lip_weight)) / (A_1 * min(C_varphi, C_weight)))
    C_pert = A_2 * max(C_weight, C_varphi) * max(Lip_weight, Lip_varphi)

    return Lemma2Variables(
        T=T_varphi * T_weight,
        S=torch.stack([T_varphi, T_weight]),
        eta=1,
        C_norm=C_norm,
        C_pert=C_pert,
        h=q,
        weight_matrices=[x for x in [point_transformation_function_weights, weight_function_weights] if x is not None],
    )


def parallel(vars1, vars2, A=1.0):
    return Lemma2Variables(
        T=torch.max(torch.max(vars1.T, vars2.T), vars1.T * vars2.T),
        S=torch.cat([vars1.S, vars2.S]),
        eta=min(vars1.eta, vars2.eta),
        C_norm=A * (vars1.C_norm + vars2.C_norm) / max(vars1.eta, vars2.eta) * max(1, vars2.C_norm / min(vars1.C_pert, vars2.C_pert)),
        C_pert=A * max(vars1.C_pert, vars2.C_pert),
        h=max(vars1.h, vars2.h),
        weight_matrices=vars1.weight_matrices + vars2.weight_matrices,
    )


def composition(vars_f, vars_g):
    # f(g(x))
    return Lemma2Variables(
        T=vars_f.T * vars_g.T,
        S=torch.cat([vars_f.S, vars_g.S]),
        eta=min(vars_f.eta, vars_g.eta),
        C_norm=vars_g.C_norm * vars_f.C_norm * max(vars_g.C_pert, vars_g.C_norm) / (
                    vars_g.C_pert * max(vars_g.eta, vars_f.eta)),
        C_pert=vars_f.C_pert * max(vars_g.C_pert, vars_g.C_norm),
        h=max(vars_f.h, vars_g.h),
        weight_matrices=vars_f.weight_matrices + vars_g.weight_matrices,
    )


def compute_bound(vars, gamma, m, delta=0.1):
    frobenius = torch.max(torch.tensor(1.0), torch.sum(torch.stack([torch.sum(matrix ** 2) for matrix in vars.weight_matrices])))

    T_squared = vars.T ** 2
    S_inv_sum_squared = (torch.sum(1 / vars.S) ** 2)
    eta_squared = vars.eta ** 2
    C_norm_squared = vars.C_norm ** 2
    log_term = torch.log(torch.tensor(vars.n * vars.h, dtype=torch.float32))

    bound = torch.sqrt(
        (frobenius * T_squared * S_inv_sum_squared * vars.h * log_term * C_norm_squared * (1 / eta_squared) +
         torch.log(torch.tensor(m / delta * max(1, 1 / vars.C_norm), dtype=torch.float32))) / (gamma ** 2) / m
    )

    return bound


def regularizer(model, m=50000, alpha=1e-4):
    gnn_params = []
    mlp_params = []
    perslay_params = []
    for param in model.named_parameters():
        if 'theta' in param[0]:
            perslay_params.append(param[1])
        if 'rho' in param[0] and 'weight' in param[0]:
            mlp_params.append(param[1])
        if 'gnn' in param[0] and 'conv' in param[0] and 'weight' in param[0]:
            gnn_params.append(param[1])
    if model.use_gnn:
        gnn_vars = from_gnn(gnn_params)
    mlp_vars = from_mlp(mlp_params)
    perslay_vars = from_gaussian_perslay(perslay_params[0])
    vars = composition(mlp_vars, parallel(gnn_vars, perslay_vars))
    factor = compute_bound(vars, gamma=1.0, m=m)
    return alpha * factor
