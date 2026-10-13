import numpy as np
def simulate_gbm(S0,mu, sigma, T, n_steps, n_paths, seed = None):
    dt = T/n_steps
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_paths, n_steps))
    X = (mu - (sigma**2)/2)*dt + sigma*np.sqrt(dt)*Z
    increments_sum = np.cumsum(X,axis = 1)
    zeros = np.zeros((n_paths,1))
    zeros_increments = np.concatenate([zeros, increments_sum], axis = 1)
    log_Sk = np.log(S0) + zeros_increments
    return np.exp(log_Sk)
     
