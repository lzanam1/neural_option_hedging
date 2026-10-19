# Deep Hedging under Transaction Costs

Neural-network hedging of a short European call, trained to minimise CVaR,
benchmarked against Black–Scholes delta hedging and an optimised no-trade band.
Based on Buehler, Gonon, Teichmann & Wood (2019), *Deep Hedging*.

## Setup
- Underlying: GBM, S0 = K = 100, σ = 20%, μ = r = 0; 30-day ATM call, daily rebalancing (N = 30)
- Proportional transaction costs c (fraction of traded notional)
- Network: MLP 3 → 32 → 32 → 1, inputs: normalised log-moneyness, time to maturity, current position;
  rolled out day by day (recurrent), trained on simulated paths
- Objective: CVaR95 via the Rockafellar–Uryasev formulation
- All results out-of-sample on 100,000 fresh paths

## Key results

**1% transaction costs:**

| Strategy | Mean P&L | Std | CVaR95 | Turnover |
|---|---|---|---|---|
| No hedge | −0.02 | 3.47 | 10.06 | 0.00 |
| Black–Scholes delta | −2.22 | 0.75 | 4.06 | 2.22 |
| No-trade band (h = 0.11, tuned on train) | −1.04 | 0.71 | 2.61 | 1.03 |
| **Deep hedging network (5 seeds)** | −1.22 ± 0.02 | **0.65 ± 0.01** | **2.40 ± 0.02** | 1.30 |

- Most of the improvement over naive BS delta comes from the no-trade principle itself.
- The network still beats the best classical band: CVaR95 −8% (gap ≈ 12× seed-to-seed std),
  std −9%, at the cost of a slightly larger mean loss — it trades a bit more to cut the tail.
- For the same CVaR95 the network-hedged option can be quoted ~26% cheaper than with BS delta.

**Sanity checks without costs:**
- The network recovers the Black–Scholes delta (mean position error 0.009, P&L std 0.369 vs 0.358).
- Trained on CVaR instead of MSE, it cuts CVaR95 by 11% vs BS delta (0.74 vs 0.83) at a higher std:
  BS delta is variance-optimal, not tail-optimal.

## Validation of the building blocks
- GBM simulator: E[S_T] and Var(log S_T) match theory within Monte Carlo error
- Black–Scholes price matches Monte Carlo; delta matches finite differences (11 digits)
- Discrete hedging error decays as 1/√N; trading costs grow as √N

## Figures
- `figures/hedging_error_vs_N.png` — 1/√N law of discrete hedging
- `figures/bs_hedging_costs.png` — risk vs cost trade-off of BS delta under costs
- `figures/pnl_bs_vs_cvar_net.png` — P&L tails, BS vs CVaR-trained network
- `figures/no_trade_band.png` — learned trading rule: partial rebalancing and no-trade region

## Limitations
- Training and testing on GBM: no fat tails, no volatility clustering
- Option premium from the frictionless BS formula; no real option quotes
- No cost for unwinding the hedge at maturity

## Next steps
- Train on block-bootstrapped historical S&P 500 returns, test on 2018–2026
- Stress windows: 2008, March 2020, 2022, April 2025
- Neural VaR/ES model for the hedged book with Kupiec backtest

## Repository
- `src/` — simulator, Black–Scholes, hedging engine (P&L, BS and band strategies, network rollout)
- `notebooks/01–06` — one notebook per stage
- `models/` — trained network weights

## Run
pip install -r requirements.txt
  
