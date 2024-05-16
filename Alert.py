#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np


# In[3]:


dataset = pd.read_csv('alert_data.csv')


# In[4]:


dataset = pd.get_dummies(dataset, columns=["Purpose/ Type of Work", "Day of the week", "Time of day", "Weather Condition"])


# In[5]:


X = dataset.drop(columns=["User Id", "Target Alert"])
y = dataset["Target Alert"]


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[7]:


rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)


# In[8]:


predictions = rf_classifier.predict(X_test)


# In[13]:


#saving the model to the disk
import pickle
pickle.dump(rf_classifier, open('Alert.pkl','wb'))  


# In[14]:


# Loading the model to compare the results
Alert = pickle.load(open('Alert.pkl','rb'))





