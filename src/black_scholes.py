import numpy as np
from scipy.stats import norm

def d1_d2(S,K,tau,sigma):
    d1 = (np.log(S/K)+ 0.5 * sigma**2 * tau)/(sigma*np.sqrt(tau))
    d2 = d1 - sigma*np.sqrt(tau)
    return d1, d2



def bs_price(S,K,tau,sigma):
    d1, d2 = d1_d2(S,K,tau,sigma)
    c = S*norm.cdf(d1)-K*norm.cdf(d2)
    return c
    

def bs_delta(S,K, tau, sigma):
    d1, d2 = d1_d2(S,K,tau,sigma)
    delta = norm.cdf(d1)
    return delta