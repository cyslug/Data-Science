
# coding: utf-8

# In[12]:


#THE IRIS DATASET IN SCIKIT-LEARN
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

iris = datasets.load_iris()


# In[13]:


type(iris)


# In[14]:


print(iris.keys())


# In[15]:


iris.target_names


# In[16]:


x = iris.data
y = iris.target
df = pd.DataFrame(x, columns=iris.feature_names)


# In[17]:


print(df.head())


# In[18]:


pd.plotting.scatter_matrix(df, c = y, figsize = [8, 8],
                 s = 150, marker = "D")


# In[ ]:




