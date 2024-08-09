#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import numpy as np


# In[2]:


# Load data
data = pd.read_csv("dataset.csv")
data.shape


# In[3]:


#Preprocessing
payment_types = data['type'].unique().tolist()
Y = data['isFraud']
X = data.drop(['isFraud', 'isFlaggedFraud', 'nameOrig', 'nameDest'], axis=1)
X['type'] = X['type'].apply(lambda x: payment_types.index(x))


# In[4]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize Random Forest Classifier
rf2 = RandomForestClassifier(n_estimators=50, random_state=42)


# In[6]:


# Train the Random Forest model
rf2.fit(X_train, Y_train)

# Make predictions
rf2_predictions = rf2.predict(X_test)


# In[7]:


# Evaluate the model
result = precision_recall_fscore_support(Y_test, rf2_predictions)
result
#               Class 0     Class 1
# Precision
# Recall
# F1 score
# Num instances

