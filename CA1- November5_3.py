#!/usr/bin/env python
# coding: utf-8

# # PCA
# 
# 

# So, to start the project, you first need to import the data set you want to analyse.
# Containing no executions yet.

# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np 
from sklearn.decomposition import PCA
import seaborn as sns; 
import matplotlib.pyplot as plt
import pandas as pd


# The next step was to read the CVS data file which is called "aps_failure_set.csv".
# Where the content of this file is stored in Data Frame (df).

# In[5]:


df = pd.read_csv("aps_failure_set.csv")


# The abbreviation for refers to the dataframe. This way you obtain more detailed information about columns and data types.

# In[6]:


df.info()


# In[7]:


df.describe()


# The code below is used to count the number of null values in each column of the data frame. This way, we can know which columns we have have null data and how many of the null values there are in each column, thus applying some of the data analysis strategies

# In[8]:


df.isnull().sum()


# In[9]:


df.head()


# In[10]:


import pandas as pd

missing_value_formats = ["n.a.","?","NA","n/a", "na", "--"]
df = pd.read_csv("aps_failure_set.csv", na_values = missing_value_formats)

def make_int(i):
    try:
        return int(i)
    except:
        return pd.np.nan 


# In[11]:


df.head()


# In[12]:


df.fillna(0,inplace=True)


# In[13]:


df.head()


# In[14]:


failure_no_label = df.drop(columns=["class"]) 
pca = PCA().fit(failure_no_label) 
plt.plot(np.cumsum(pca.explained_variance_ratio_)) 
plt.xlabel('number of components') 
plt.ylabel('cumulative explained variance');


# In[15]:


pca = PCA(10)
projected = pca.fit_transform(failure_no_label)


# In[16]:


projected.shape


# In[17]:


newfailure_df = pd.DataFrame(projected, columns =['C1', 'C2', 'C3', 'C4','C5', 'C6', 'C7', 'C8', 'C9','C10'])


# In[18]:


newfailure_df.head()


# In[19]:


newfailure_df["class"] = df["class"]


# In[20]:


newfailure_df.head() 


# In[21]:


df_array=newfailure_df.values 


# In[23]:


from sklearn.model_selection import train_test_split
X = df_array[:,0:10]
y = df_array[:,10]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)


# In[25]:


from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


# In[26]:


models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))


# In[ ]:


results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))


# In[ ]:




