#!/usr/bin/env python
# coding: utf-8

# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


# In[15]:


df = pd.read_csv("aps_failure_set.csv")


# In[17]:


df.info()


# In[18]:


df.describe()


# In[21]:


df.head()


# In[ ]:




