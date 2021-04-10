#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[2]:


directory = '../dataset/'
stocks = ['AMD.csv', 'FB.csv', 'TSLA.csv', 'TWTR.csv', 'MONDY.csv']
stocks = [directory + s for s in stocks]
stocks


# In[3]:


dfs = [pd.read_csv(s)[['Date', 'Close']] for s in stocks]


# In[4]:


from functools import reduce
data = reduce(lambda left,right: pd.merge(left,right,on='Date'), dfs).iloc[:, 1:]
data.head()


# In[6]:


returns = data.pct_change()
mean_daily_returns = returns.mean()
cov_matrix = returns.cov()


# In[8]:


cov_matrix


# In[9]:


num_portfolios = 25000
results = np.zeros((3,num_portfolios))


# In[11]:


for i in range(num_portfolios):
    weights = np.random.random(cov_matrix.shape[0])
    weights /= np.sum(weights)
    portfolio_return = np.sum(mean_daily_returns * weights) * 252
    portfolio_std_dev = np.sqrt(np.dot(weights.T,np.dot(cov_matrix, weights))) * np.sqrt(252)
    results[0,i] = portfolio_return
    results[1,i] = portfolio_std_dev
    results[2,i] = results[0,i] / results[1,i]


# In[12]:


results_frame = pd.DataFrame(results.T,columns=['ret','stdev','sharpe'])


# In[18]:


plt.figure(figsize = (7, 5))
plt.scatter(results_frame.stdev,results_frame.ret,c=results_frame.sharpe,cmap='RdYlBu')
plt.colorbar()
plt.xlabel('volatility')
plt.ylabel('returns')
plt.show()


# In[ ]:




