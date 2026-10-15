import numpy as np

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