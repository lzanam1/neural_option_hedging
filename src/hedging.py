import numpy as np
from src.black_scholes import bs_delta
import torch

def compute_pnl(S, deltas, p0, K, c=0.0):
    payoff = np.maximum(0, S[:, -1] - K)
    daily_gains = (S[:, 1:] - S[:, :-1]) * deltas
    sum_daily_gains = daily_gains.sum(axis=1)
    zeros = np.zeros((deltas.shape[0], 1))
    prev_deltas = np.concatenate([zeros, deltas[:, :-1]], axis=1)
    trades = np.abs(prev_deltas - deltas)
    costs = S[:, :-1] * c * trades
    sum_costs = costs.sum(axis=1)
    return p0 - payoff + sum_daily_gains - sum_costs


def bs_delta_strategy(S, K, T, sigma):
    n_steps = S.shape[1] - 1
    dt = T / n_steps
    tau = T - np.arange(n_steps) * dt
    deltas = bs_delta(S[:, :-1], K, tau, sigma)
    return deltas


def rollout(net, x1_b, x2_b):
    B, n_steps = x1_b.shape
    delta_prev = torch.zeros(B)
    deltas_list = []

    for k in range(n_steps):
        inp = torch.stack([x1_b[:, k], x2_b[:, k], delta_prev], dim=-1)
        delta_k = net(inp).squeeze(-1)
        deltas_list.append(delta_k)
        delta_prev = delta_k

    return torch.stack(deltas_list, dim=1)


def pnl_torch(S_b, deltas, p0, K, c):
    payoff = torch.clamp(S_b[:, -1] - K, min=0)
    hedge_gain = (deltas * (S_b[:, 1:] - S_b[:, :-1])).sum(dim=1)

    zeros = torch.zeros(deltas.shape[0], 1)
    prev = torch.cat([zeros, deltas[:, :-1]], dim=1)
    costs = (c * S_b[:, :-1] * torch.abs(deltas - prev)).sum(dim=1)

    return p0 - payoff + hedge_gain - costs