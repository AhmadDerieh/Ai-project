#!/usr/bin/env python
# coding: utf-8

# In[21]:


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans as kmeans
import numpy as np


# In[22]:


import csv
def readFileThroughCSV(filename):
    csvfile = open(filename)
    readerobject = csv.reader(csvfile, delimiter=',')
    lst = list(readerobject)
    csvfile.close()

    dataX = [x[1] for x in lst[1:]]
    dataY = [x[2] for x in lst[1:]]

    arrX = np.array(dataX)
    arrX = arrX.astype(float)
    arrY = np.array(dataY)
    arrY = arrY.astype(float)

    arr = np.array([arrX,arrY])
    arr = np.transpose(arr)
    return(arr)


# In[23]:


arr = readFileThroughCSV("ai_project.csv")


# In[24]:


kmeans_model = kmeans(n_clusters=3)
kmeans_model.fit(arr)

lab = kmeans_model.labels_

centroid = kmeans_model.cluster_centers_

inertia = kmeans_model.inertia_


# In[18]:


import pandas as pd
import glob
import numpy as np
diction = {
    "X": arr[:,0],
    "Y": arr[:,1],
    "Labels": lab
}

df = pd.DataFrame(data=diction)
df.to_csv('ClusteringResults .csv')

