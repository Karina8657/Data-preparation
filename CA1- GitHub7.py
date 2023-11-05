#!/usr/bin/env python
# coding: utf-8

# ## Karina Fracieli Schmidt - 2023314
# 
# CA1 - Data Preparation

# # PCA
# 
#     PCA (Principal Component Analysis) serves as a vital data analysis technique in Python, facilitating dimensionality reduction, the detection of concealed patterns, exploratory data analysis, data preprocessing, and visualization in lower-dimensional spaces, aiding in the comprehension of variable relationships. Widely applied in data science and data analysis to tackle high-dimensional datasets, PCA simplifies intricate analyses and fosters an effective grasp of data structures.

# ## Reding data
# 

# To star the project, first starts with the codes below and are used to import and visualize data.
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

The following code will generate a statistical summary of the data frame. Calculating the count, mean, standard deviation, minimum, quartiles and maximum for each numeric column in the DataFrame. 

## Count: shows the number of non-missing observations in the aa_000 column, which indicates 60,000 valid observations in that column.

## Mean: The mean represents the average value of all numbers in the "aa_000" column, which is approximately 59,336.5. 

## Std: Standard deviation (std) measures the spread of values in the "aa_000" column. The standard deviation is approximately 145,430.1, this indicates that the values are dispersed in relation to the mean. 

## Min: This is the minimum value in the "aa_000" column. The minimum value is 0, so the lowest value that was found in this column was 0.

## 25%: It shows that 25% are below and 75% are above. In this project, 25% is approximately 834.

## 50%: The median is the middle value, which means half of the values are below and the other half are above. 

##75% : It is the value at which 75% of the data is below and 25% is above.

## MAx: This is the maximum value, in this case is approximately 2,746,564.

 
# In[5]:


df.describe()


# The code below is used to count the number of null values in each column of the data frame. This way, we can know which columns we have have null data and how many of the null values there are in each column, thus applying some of the data analysis strategies.
# 
# The output information is:
# 
# Lenght in this case is 171 elements and each element represents 1 column, so we have 171 columns.
# The dtype information indicates the data type of the count values, which are 64-bit integers.

# In[6]:


df.isnull().sum()


# In[7]:


df.head()


# We can see from the table above that there were clearly missing values. In the present case, we employ the subsequent code to perform the conversion of this missing data, which is generally interpreted by pandas as NaN (not as a number). Therefore, we can now see it in the table below.

# In[8]:


import pandas as pd

missing_value_formats = ["n.a.","?","NA","n/a", "na", "--"]
df = pd.read_csv("aps_failure_set.csv", na_values = missing_value_formats)

def make_int(i):
    try:
        return int(i)
    except:
        return pd.np.nan 


# So, after applying the code, it is now possible to see the missing values change to NaN.

# In[9]:


df.head()


# To handle missing values, the code below will transform these values from NaN to 0 in the data frame in question. Modifications are made to the pre-existing data frame, without creating a new data frame.

# In[10]:


df.fillna(0,inplace=True)


# After applying the code, you will notice the substitution of zero where no data is missing.

# In[11]:


df.head()


# The following code is used to apply Principal Component Analysis (PCA) to a Data Frame.
# From the generated graph, it will be possible to view the cumulative variance. With this information, it will be possible to make a decision on how many components will be needed to maintain data variation.
# 

# In[28]:


failure_no_label = df.drop(columns=["class"]) 
pca = PCA().fit(failure_no_label) 
plt.plot(np.cumsum(pca.explained_variance_ratio_)) 
plt.xlabel('number of components') 
plt.ylabel('cumulative explained variance');


# By looking at the Cumulative Explained Variance graph above, we can determine the required number of principal components to maintain the variance. When the curve remains constant, it is possible to maintain a sufficient number of principal components to retain a high percentage of the total variance. This results in a more compact and informative representation of the data while reducing dimensionality. In the present scenario, 10 main components were used. It is noted that, even with the discarding of several components, it was possible to preserve a variance of at least 95%.
# 

# The next step consists of applying PCA, which will reduce the existing data set, while maintaining its variation. Now, the data frame has only 10 components after a reduction of 171. This way, the data that will be analyzed is simplified.

# In[29]:


pca = PCA(10)
projected = pca.fit_transform(failure_no_label)


# Below, we have the new number of rows and columns, after applying PCA.

# In[14]:


projected.shape


# Now, it is possible to create a new data frame with the name newfailure. Additionally, we have the new names for the columns that were created and renamed as C1, C2, C3, C4, C5, C6, C7, C8, C9 and C10.

# In[15]:


newfailure_df = pd.DataFrame(projected, columns =['C1', 'C2', 'C3', 'C4','C5', 'C6', 'C7', 'C8', 'C9','C10'])


# In[16]:


newfailure_df.head()


# Just copying the class column to the new data frame, as we can see the "class" column at the end of the table.

# In[17]:


newfailure_df["class"] = df["class"]


# In[18]:


newfailure_df.head() 


# The array is a compound variable that allows us to work with storing multiple values.
# It is being created to store the data frame values.

# In[19]:


df_array=newfailure_df.values 


# The following codes aim to divide a dataset into two distinct parts: a training set and a validation set. To perform this division, they use the train_test_split function. Data set attributes are stored in X and labels are stored in y. Then they are separated into groups. This set aims to evaluate the performance of machine learning models when testing them with unknown data, making it possible to analyze their performance in these circumstances.
# To ensure that there is a validation data set, we will reserve 20% of the data with the code test_size=0.20, while the remaining 80% will be used for training.

# In[20]:


from sklearn.model_selection import train_test_split
X = df_array[:,0:10]
y = df_array[:,10]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)


# O sklearn é um biblioteca de códigos bastante utilizada em Python, entre suas funções análise de dados.
# Foi realizada a diferentes importações para o uso de diferentes funções de machine learn. 
# Dessa forma, criando e deselvolvendo resultados nos códigos Python.
# 
# 
# 
# 
# 

# In[21]:


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


# In the next step, the code is generating a list of machine learning models that employ distinct algorithms and configurations. Each item in the list consists of a short name for the model, followed by an instance of that model containing specific parameters. These models can later be leveraged to improve and evaluate performance on machine learning-related tasks.

# In[22]:


models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))


# By using the previously mentioned import and implementing the following codes, we obtain the printout of the mean and standard deviation, which represents the measure of variability.
# By observing the results obtained, we can select the model that most accurately adapts to the data we are analyzing.
# The numbers demonstrate that the highest averages were obtained by LDA, KNN and SVM, but their variability was lower.
# Logistic regression (LR) was the one that had the best standard deviation value (variability).

# In[23]:


results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))


# From the results generated above, we can now create a graph to better visualize the distribution, median and performances of the different models previously applied. The graph can also show outliers.
# This makes it easier to choose the model that best fits the data.
# 
# 
# 
# 

# In[24]:


pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()


# In[25]:


model = LogisticRegression(solver='liblinear', multi_class='ovr')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)


# In[26]:


print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))


# ## Curse of Dimentiolaly
# 
# it is challenges that arise when dealing with high-dimensional datasets in data analytics and machine learning. As the number of variables or dimensions increases, complexity and missing data increase, making analysis and interpretation more difficult. This also leads to difficulties in visualization, increased computational complexity, and reduced ability to discriminate patterns in the data. To mitigate this issue, it is essential to use dimensionality reduction techniques and appropriate data preparation strategies.
# 
# The Curse of Dimensionality is a critical challenge in high-dimensional data analysis, due to increased complexity, data scarcity, and difficulty in visualization. Dimensionality reduction and judicious choice of data preparation techniques are fundamental to dealing with complex data sets and ensuring effective results in data analysis and machine learning.

# In[ ]:




