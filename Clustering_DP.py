#!/usr/bin/env python
# coding: utf-8

# In[24]:


# importPackages

import pandas as pd
import pandas_profiling

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

from yellowbrick.cluster import SilhouetteVisualizer, InterclusterDistance, KElbowVisualizer

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"



#--------------------------------------------------------------------------------------------
# Data infile
#--------------------------------------------------------------------------------------------

# Change source path + file as desired
path = ""
file = "jewelry_customers.csv"
df = pd.read_csv(path + file)

#--------------------------------------------------------------------------------------------
# Data Exploration & Analysis
#--------------------------------------------------------------------------------------------

list(df)# list attribute names
dl = list(df) 
df.shape # instances, attributes
df.info() # data types
df.describe().transpose() # df attribute stats

# Check for nulls
dfNull = df.isnull().values.any().sum().sum()
print (df.isna())
print ('Checking Nulls: # of Obs')
print (dfNull)

# Check for duplicates
dfDup = df.duplicated().values.any().sum().sum()
print ('Checking Dups: # of Obs')
print (dfDup)


#correlationMatrix
df2 = pd.DataFrame(df,columns=dl)


corrMatrix = df2.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()


# In[ ]:


# Appears Income & SpendingScore are correlated


# In[26]:


#--------------------------------------------------------------------------------------------
# DATA PREPROCESSING
#--------------------------------------------------------------------------------------------
# sort spendingScore by descending order
df.sort_values(by=['SpendingScore'], inplace=True, ascending=False)
#df.["Age"] = pd.to_object(df["Age"])
df.head(n=10) 
df.tail(n=10)


# In[27]:


# Plot and check distribution
sns.jointplot(x="Income", y="SpendingScore", data=df);


# # Standardize the Data

# In[28]:


# Keep only correlated features
#X = df.drop(['Age','Savings'],axis=1)
X = df.copy()

#--------------------------------------------------------------------------------------------
# Perform Scaling
#--------------------------------------------------------------------------------------------
scaler = StandardScaler()
features = ['Age', 'Income', 'SpendingScore', 'Savings']
X[features] = scaler.fit_transform(X[features])


# In[29]:


X.shape
X.info()
X.describe().transpose()
X.head(10)


# In[30]:


# Checking plots after scaling
sns.jointplot(x="Income", y="SpendingScore", data=X);


# In[31]:

#--------------------------------------------------------------------------------------------
# K-MEANS CLUSTERING
# FOR LOOP --> TRAIL & ERROR of K = 2 to 11, Measurement of Inertia and Silhoutte
#--------------------------------------------------------------------------------------------

from sklearn.cluster import KMeans
wcss = []
ss = []
for i in range(2, 11):
        print (i)    
        k_means = KMeans(init='k-means++', n_clusters=i, n_init=10, random_state=42)
        k_means.fit(X)
        print ('WCSS - Inertia')
        k_means.inertia_
        wcss.append(k_means.inertia_)
        
        print ('silhouette_score')
        k_means.labels_
        silhouette_score(X, k_means.labels_)
        ss.append(silhouette_score(X, k_means.labels_))      


# In[32]:

#--------------------------------------------------------------------------------------------
# CHECK OUTPUTS
#--------------------------------------------------------------------------------------------

# Evaluate Measurement of Inertia and Silhoutte
print ('WCSS')
wcss

print ('Silhouette_Score')
ss


# In[33]:


# Setting K = 5
k_means = KMeans(init='k-means++', n_clusters=5, n_init=10, random_state=42)
k_means.fit(X)


# In[34]:


k_means.labels_


# In[35]:


# Check the centers of clusters
k_means.cluster_centers_


# In[36]:

#--------------------------------------------------------------------------------------------
# PLOT
#--------------------------------------------------------------------------------------------

plt.style.use('default');

plt.figure(figsize=(16, 10));
plt.grid(True);

sc = plt.scatter(X.iloc[:, 0], X.iloc[:, 1], s=200, c=k_means.labels_);
#plt.scatter(k_means.cluster_centers_[:, 0], k_means.cluster_centers_[:, 1], marker='x', s=500, c="black")
plt.title("K-Means (K=5)", fontsize=20);
plt.xlabel('Income (K)', fontsize=22);
plt.ylabel('Spending Score', fontsize=22);
plt.xticks(fontsize=18);
plt.yticks(fontsize=18);

for label in k_means.labels_:
    plt.text(x=k_means.cluster_centers_[label, 0], y=k_means.cluster_centers_[label, 1], s=label, fontsize=32, 
             horizontalalignment='center', verticalalignment='center', color='black',
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.1', alpha=0.02));


# # Internal Validation Metrics

# In[37]:


# WCSS == Inertia
print ('WCSS - Inertia')
k_means.inertia_

print ('silhouette_score')
silhouette_score(X, k_means.labels_)


# In[38]:


plt.style.use('default');

sample_silhouette_values = silhouette_samples(X, k_means.labels_)
sizes = 200*sample_silhouette_values

plt.figure(figsize=(16, 10));
plt.grid(True);

plt.scatter(X.iloc[:, 0], X.iloc[:, 1], s=sizes, c=k_means.labels_)
plt.scatter(k_means.cluster_centers_[:, 0], k_means.cluster_centers_[:, 1], marker='x', s=500, c="black")

plt.title("K-Means (Dot Size = Silhouette Distance)", fontsize=20);
plt.xlabel('Income (K)', fontsize=22);
plt.ylabel('Spending Score', fontsize=22);
plt.xticks(fontsize=18);
plt.yticks(fontsize=18);


# In[39]:


visualizer = SilhouetteVisualizer(k_means)
visualizer.fit(X)
visualizer.poof()
fig = visualizer.ax.get_figure()


# In[40]:


# Instantiate the clustering model and visualizer
visualizer = InterclusterDistance(k_means)

visualizer.fit(X) # Fit the training data to the visualizer
visualizer.poof() # Draw/show/poof the data
#plt.savefig('out/mall-kmeans-5-tsne.png', transparent=False);


# In[41]:


inertias = {}
silhouettes = {}
for k in range(2, 11):
    kmeans = KMeans(init='k-means++', n_init=10, n_clusters=k, max_iter=1000, random_state=42).fit(X)
    inertias[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
    silhouettes[k] = silhouette_score(X, kmeans.labels_, metric='euclidean')
    

plt.figure();
plt.grid(True);
plt.plot(list(inertias.keys()), list(inertias.values()));
plt.title('K-Means, Elbow Method')
plt.xlabel("Number of clusters, K");
plt.ylabel("Inertia");
#plt.savefig('out/mall-kmeans-elbow-interia.png');


plt.figure();
plt.grid(True);
plt.plot(list(silhouettes.keys()), list(silhouettes.values()));
plt.title('K-Means, Elbow Method')
plt.xlabel("Number of clusters, K");
plt.ylabel("Silhouette");

#--------------------------------------------------------------------------------------------
# Interpreting the Clusters
#--------------------------------------------------------------------------------------------

# In[42]:


print ('Means')
k_means.cluster_centers_

print ('Inverse Transofrm Features')
# Inverse transform to see true centers
scaler.inverse_transform(k_means.cluster_centers_)


# In[43]:


#---------------------------------------------------------------
# CLUSTER SUMMARY STATISTICS - PRINT
#---------------------------------------------------------------

#Viewing each cluster and its statistics

from scipy import stats
import seaborn as sns

pd.set_option("display.precision", 2)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

print('All Data:')
print('Number of Instances: {}'.format(X.shape[0]))
df.describe().transpose()

for col in df:
    df[col].value_counts()

#for label in set(k_means.labels_):
 #   print('\nCluster {}:'.format(label))
    
for i, label in enumerate(set(k_means.labels_)):
    n = df.iloc[k_means.labels_==label].shape[0]
                
    print('\nCluster {}:'.format(label))
    print('Number of Instances: {}'.format(n))

    df.iloc[k_means.labels_==label].describe().transpose()
        
    for col in df:
        df.iloc[k_means.labels_==label][col].value_counts()



#---------------------------------------------------------------
# Find Examplars for Personas
#---------------------------------------------------------------
from scipy.spatial import distance

for i, label in enumerate(set(k_means.labels_)):    
    X_tmp = X[k_means.labels_==label].copy()
    
    exemplar_idx = distance.cdist([k_means.cluster_centers_[i]], X_tmp).argmin()
    #exemplar = pd.DataFrame(X_tmp.iloc[exemplar_idx])
    exemplar = pd.DataFrame(X_tmp.iloc[exemplar_idx])
    #test = scaler.inverse_transform(exemplar)
    print('\nCluster {}:'.format(label))
    exemplar
    


# In[45]:


#--------------------------------------------------------------------------------------------
# Sample Dataset for testing
#--------------------------------------------------------------------------------------------

import numpy as np
chosen_idx = np.random.choice(505, replace=False, size=12)
test = df.iloc[chosen_idx]
test.head()


# In[46]:


#--------------------------------------------------------------------------------------------
# Perform Scaling for Test Set
#--------------------------------------------------------------------------------------------
scaler = StandardScaler()
features = ['Age', 'Income', 'SpendingScore', 'Savings']
test[features] = scaler.fit_transform(test[features])


# In[47]:


#--------------------------------------------------------------------------------------------
# Predict
#--------------------------------------------------------------------------------------------
kmeans.fit(test)
pred = kmeans.predict(test)


# In[48]:


#--------------------------------------------------------------------------------------------
# Results
#--------------------------------------------------------------------------------------------
frame = pd.DataFrame(test)
frame['cluster'] = pred
frame['cluster'].value_counts()

frame.head()

