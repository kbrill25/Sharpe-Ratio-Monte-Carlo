# -*- coding: utf-8 -*-
"""
Sharpe Ratio and Monte Carlo Methods
Credit to https://blog.quantinsti.com/portfolio-optimization-maximum-return-risk-ratio-python/
I largely derived my inspiration from Quantinsti.com and need to give proper credit
"""
#import necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as webdata

#in this case, I will be evaluate semiconductor/IC companies
semiconductor_tickers = ['AMD', 'INTC', 'AMAT', 'NVDA']

#get the data using webdata and extract Adjusted Close data
semiconductor_stock_data = webdata.DataReader(semiconductor_tickers, data_source="yahoo",start="01/01/2020",end="05/05/2021")
semi_adj = semiconductor_stock_data['Adj Close']

#rearrange
semi_adj = semi_adj.iloc[::-1]

#for sharpe ratio, we need the return of the portfolio
#calculate returns
semi_returns = semi_adj.pct_change()

#covariance matrix
semi_covariance = semi_returns.cov()
print(semi_covariance)

#mean return
mean_semi_returns = semi_returns.mean()
print(mean_semi_returns)

#now we can begin sharpe ratio 
iterations = 2500   #arbitrary, can customize
semi_length = 4 + len(semiconductor_tickers) - 1
simulation = np.zeros((semi_length, iterations))

#looping to generate weights in calculation
for i in range(iterations):
    weights = np.array(np.random.random(4))
    weights = weights/np.sum(weights)
   
    #evaluating the portfolio
    port_mean_ret = np.sum(mean_semi_returns * weights)
    port_sd_ret = np.sqrt(np.dot(weights.T,np.dot(semi_covariance, weights)))
    
    simulation[0,i] = port_mean_ret
    simulation[1,i] = port_sd_ret
    
    #sharpe ratio
    simulation[2,i] = simulation[0,i]/simulation[1,i]
    
    for j in range(len(weights)):
        simulation[j+3,i] = weights[j]
    simulation_data = pd.DataFrame(simulation.T,columns=['Returns','SD','Sharpe','AMD','INTC','AMAT','NVDA'])
    
#generate visualizations
plt.scatter(simulation_data.SD,simulation_data.Returns,c=simulation_data.Sharpe,cmap="winter")
plt.xlabel('Standard Deviation')
plt.ylabel('Returns')
plt.show()

