#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import pandas as pd


# ## Let say
# 
# Let say, TWTR.csv is my realtime data (follow [realtime-evolution-strategy.ipynb](realtime-evolution-strategy.ipynb)), remember, we trained using `Close`, and `Volume` data.
# 
# So every request means new daily data.
# 
# You can improve the code to bind historical data with your own database or any websocket streaming data. Imagination is your limit now.

# In[2]:


df = pd.read_csv('TWTR.csv')
df.head()


# In[3]:


close = df['Close'].tolist()
volume = df['Volume'].tolist()


# ## Check balance

# In[4]:


requests.get('http://localhost:8005/balance').json()


# This is the initial capital we have for now, you can check [agent.ipynb](https://github.com/huseinzol05/Stock-Prediction-Models/blob/master/realtime-agent/agent.ipynb) how I defined it, or you can overwrite it.

# ## Trading

# In[5]:


import json

data = json.dumps([close[0], volume[0]])
data


# Remember, my last training session was only used `Close` and `Volume`, you need to edit it to accept any kind of parameters.

# In[6]:


requests.get('http://localhost:8005/trade?data='+data).json()


# Reason why you got 'data not enough to trade', because, the agent waiting another data to complete the queue, atleast same as `window_size` size.
# 
# Last time I defined `window_size` is 20, means, it only look back 20 historical data to trade.

# Assume now, you have 100 times new datapoints going in, you want to trade these datapoints.

# In[7]:


for i in range(200):
    data = json.dumps([close[i], volume[i]])
    requested = requests.get('http://localhost:8005/trade?data=' + data).json()
    print(requested)


# In[ ]:




