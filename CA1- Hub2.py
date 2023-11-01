#!/usr/bin/env python
# coding: utf-8

# # PCA
# 
# 

# So, to start the project, you first need to import the data set you want to analyse.
# Containing no executions yet.

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np 
from sklearn.decomposition import PCA
import seaborn as sns; 
import matplotlib.pyplot as plt
import pandas as pd


# The next step was to read the CVS data file which is called "aps_failure_set.csv".
# Where the content of this file is stored in Data Frame (df).

# In[3]:


df = pd.read_csv("aps_failure_set.csv")


# The abbreviation for refers to the dataframe. This way you obtain more detailed information about columns and data types.

# In[4]:


df.info()


# In[5]:


df.describe()


# The code below is used to count the number of null values in each column of the data frame. This way, we can know which columns we have have null data and how many of the null values there are in each column, thus applying some of the data analysis strategies

# In[7]:


df.isnull().sum()


# In[8]:


df.head()


# In[9]:


import pandas as pd

missing_value_formats = ["n.a.","?","NA","n/a", "na", "--"]
df = pd.read_csv("aps_failure_set.csv", na_values = missing_value_formats)

def make_int(i):
    try:
        return int(i)
    except:
        return pd.np.nan 


# In[10]:


df.head()


# In[11]:


df.fillna(0,inplace=True)


# In[12]:


df.head()


# In[ ]:




