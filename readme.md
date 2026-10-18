# Deep Hedging

Neural network hedging of European options under transaction costs,
benchmarked against Black–Scholes delta hedging.

## Status
- [x] Stage 1: GBM price simulator with validation against theory
- [x] Stage 2: Black–Scholes pricing and delta
- [x] Stage 3: Hedging engine and benchmark
- [x] Stage 4+: Neural network hedging
- [x] Stage 5: Training on CVaR95 instead of MSE. Without costs.

# Stage 3 findings - Daily Black–Scholes delta hedging reduces P&L std from 3.46 (unhedged) to 0.36.
- Without costs, hedging error decays as 1/sqrt(N) with the number of rebalancing dates.
- With proportional costs, trading costs grow as sqrt(N): at c = 1% and N = 100 the mean
  loss (−3.68) exceeds the option premium (2.29), and risk starts increasing again.
- BS delta ignores costs and trades identically for any c and this is the gap
  the neural network should close.
# Stage 4 findings - Neural network hedger (MLP, 1185 params) trained only on simulated
  paths recovers the Black–Scholes delta: mean position error 0.009,
  out-of-sample P&L std 0.369 vs 0.358 for BS delta.
# Stage 5 findings - the CVaR-trained network cuts CVaR95 by 11% vs BS delta (0.739 vs 0.833, out-of-sample),
  at the cost of higher P&L std (0.409 vs 0.358).

  
