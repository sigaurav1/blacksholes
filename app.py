import numpy as np
from scipy.stats import norm
from fastapi import FastAPI
from pydantic import BaseModel
from math import sqrt
from typing import Optional
# Define Model Variables
# Description based on National Stock Exchange, India
# r = 0.052 # Rate of interest is the relevant MIBOR 
# S = 16220
# K = 16200
# T = 4/365
# sigma = 0.181

app = FastAPI()
class blackScholes(BaseModel):

    r : float
    S : int
    K : int
    T : float
    marketPrice : float
    sigma : Optional[float] = 0.1
    type : str

# Implementing black scholes model
@app.get('/')
def echo():
    return {'message':'echo'}

def d1d2Computation(item):
    d1 = (np.log(item.S/item.K) + (item.r + item.sigma**2/2)*item.T)/(item.sigma * np.sqrt(item.T))
    d2 = d1 - item.sigma*np.sqrt(item.T)
    return d1, d2

@app.post('/price')
def blackScholesPrice(item:blackScholes):
    # d1 = (np.log(item.S/item.K) + (item.r + item.sigma**2/2)*item.T)/(item.sigma * np.sqrt(item.T))
    # d2 = d1 - item.sigma*np.sqrt(item.T)
    d1, d2 = d1d2Computation(item)
    try:
        if item.type == 'C':
            optionPrice = item.S * norm.cdf(d1, 0, 1) - item.K*np.exp(-item.r*item.T)*norm.cdf(d2, 0, 1)
            return optionPrice
        elif item.type == 'P':
            optionPrice = item.K*np.exp(-item.r*item.T)*norm.cdf(-d2, 0, 1) - item.S * norm.cdf(-d1, 0, 1)
            return optionPrice
    except Exception as e:
        print(e)   

# Calculating IV using the Newton Raphson Method
@app.post('/iv')
def blackScholesIV(item:blackScholes):
    max_try = 1000
    d1, d2 = d1d2Computation(item)
    item.sigma = 0.01
    for i in range(max_try):
        theoreticalPrice = blackScholesPrice(item)
        diff = item.marketPrice - theoreticalPrice
        N_Price = norm.pdf
        vega = item.S*N_Price(d1)*sqrt(item.T)
        if abs(diff) < 0.0001:
            return item.sigma
        item.sigma += diff/vega
        print(theoreticalPrice)
    return item.sigma