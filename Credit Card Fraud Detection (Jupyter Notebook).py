#!/usr/bin/env python
# coding: utf-8

# ## Author: Sadaf Shaikh

# # Credit Card Fraud Detection 
# 
# 
# Throughout the financial sector, machine learning algorithms are being developed to detect fraudulent transactions.  In this project, that is exactly what we are going to be doing as well.  Using a dataset of of nearly 28,500 credit card transactions and multiple unsupervised anomaly detection algorithms, we are going to identify transactions with a high probability of being credit card fraud.  In this project, we will build and deploy the following two machine learning algorithms:
# 
# * Local Outlier Factor (LOF)
# * Isolation Forest Algorithm
# 
# Furthermore, using metrics suchs as precision, recall, and F1-scores, we will investigate why the classification accuracy for these algorithms can be misleading.
# 
# In addition, we will explore the use of data visualization techniques common in data science, such as parameter histograms and correlation matrices, to gain a better understanding of the underlying distribution of data in our data set. Let's get started!
# 
# ## 1. Importing Necessary Libraries
# 
# To start, let's print out the version numbers of all the libraries we will be using in this project. This serves two purposes - it ensures we have installed the libraries correctly and ensures that this tutorial will be reproducible. 
# 

# In[1]:


import sys
import numpy
import pandas
import matplotlib
import seaborn
import scipy

print('Python: {}'.format(sys.version))
print('Numpy: {}'.format(numpy.__version__))
print('Pandas: {}'.format(pandas.__version__))
print('Matplotlib: {}'.format(matplotlib.__version__))
print('Seaborn: {}'.format(seaborn.__version__))
print('Scipy: {}'.format(scipy.__version__))


# In[2]:


# import the necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ### 2. The Data Set
# 
# In the following cells, we will import our dataset from a .csv file as a Pandas DataFrame.  Furthermore, we will begin exploring the dataset to gain an understanding of the type, quantity, and distribution of data in our dataset.  For this purpose, we will use Pandas' built-in describe feature, as well as parameter histograms and a correlation matrix. 

# In[4]:


# Load the dataset from the csv file using pandas
data = pd.read_csv('creditcard.csv')


# In[5]:


# Start exploring the dataset
print(data.columns)


# In[6]:


# Print the shape of the data
data = data.sample(frac=0.1, random_state = 1)
print(data.shape)
print(data.describe())

# V1 - V28 are the results of a PCA Dimensionality reduction to protect user identities and sensitive features


# In[7]:


# Plot histograms of each parameter 
data.hist(figsize = (20, 20))
plt.show()


# In[8]:


# Determine number of fraud cases in dataset

Fraud = data[data['Class'] == 1]
Valid = data[data['Class'] == 0]

outlier_fraction = len(Fraud)/float(len(Valid))
print(outlier_fraction)

print('Fraud Cases: {}'.format(len(data[data['Class'] == 1])))
print('Valid Transactions: {}'.format(len(data[data['Class'] == 0])))


# In[9]:


# Correlation matrix
corrmat = data.corr()
fig = plt.figure(figsize = (12, 9))

sns.heatmap(corrmat, vmax = .8, square = True)
plt.show()


# In[10]:


# Get all the columns from the dataFrame
columns = data.columns.tolist()

# Filter the columns to remove data we do not want
columns = [c for c in columns if c not in ["Class"]]

# Store the variable we'll be predicting on
target = "Class"

X = data[columns]
Y = data[target]

# Print shapes
print(X.shape)
print(Y.shape)


# ## 3. Unsupervised Outlier Detection
# 
# Now that we have processed our data, we can begin deploying our machine learning algorithms.  We will use the following techniques: 
# 
# **Local Outlier Factor (LOF)**
# 
# The anomaly score of each sample is called Local Outlier Factor. It measures the local deviation of density of a 
# given sample with respect to its neighbors. It is local in that the anomaly score depends on how isolated the 
# object is with respect to the surrounding neighborhood.
# 
# 
# **Isolation Forest Algorithm**
# 
# The IsolationForest ‘isolates’ observations by randomly selecting a feature and then randomly selecting 
# a split value between the maximum and minimum values of the selected feature.
# 
# Since recursive partitioning can be represented by a tree structure, the number of splittings required to 
# isolate a sample is equivalent to the path length from the root node to the terminating node.
# 
# This path length, averaged over a forest of such random trees, is a measure of normality and our decision function.
# 
# Random partitioning produces noticeably shorter paths for anomalies. Hence, when a forest of random trees 
# collectively produce shorter path lengths for particular samples, they are highly likely to be anomalies.

# In[11]:


from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# define random states
state = 1

# define outlier detection tools to be compared
classifiers = {
    "Isolation Forest": IsolationForest(max_samples=len(X),
                                        contamination=outlier_fraction,
                                        random_state=state),
    "Local Outlier Factor": LocalOutlierFactor(
        n_neighbors=20,
        contamination=outlier_fraction)}


# In[12]:


# Fit the model
plt.figure(figsize=(9, 7))
n_outliers = len(Fraud)


for i, (clf_name, clf) in enumerate(classifiers.items()):
    
    # fit the data and tag outliers
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)
    
    # Reshape the prediction values to 0 for valid, 1 for fraud. 
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    
    n_errors = (y_pred != Y).sum()
    
    # Run classification metrics
    print('{}: {}'.format(clf_name, n_errors))
    print(accuracy_score(Y, y_pred))
    print(classification_report(Y, y_pred))


# In[ ]:




