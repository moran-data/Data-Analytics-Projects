#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from PIL import Image

np.random.seed(42)
from sklearn.metrics import confusion_matrix

import keras
from keras.utils.np_utils import to_categorical # used for converting labels to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.preprocessing import LabelEncoder


# In[3]:


df = pd.read_csv(r"C:\Users\User\Downloads\ISIC_2019_Training_GroundTruth.csv")


# Types of malignant cancer 
# We would gorup the types to 3 Categories - 
# 
#     1.Melanoma
#     
#     2.Basal cell carcinoma
#   
#     Type 3 - could be benign or curable in most of the cases
#     3.  #Actinic keratosis -  The patches are not usually serious / cancer
#         # Dermatofibroma -  Dermatofibromas are referred to as benign fibrous 
#         #Vascular lesion - may be benign (not cancer) or malignant (cancer) 
#         #Squamous cell carcinoma - In general, the squamous cell carcinoma survival rate is very high
# 
# The type below are benign and were considered in the first model       
# -Melanocytic nevus
# -Benign keratosis (solar lentigo / seborrheic keratosis / lichen planus-like keratosis)

# In[4]:


# Labeling the types 1 - 6

df['BCC'] = df['BCC'].replace(1.0, 2)
df['AK'] = df['AK'].replace(1.0, 3)
df['DF'] = df['DF'].replace(1.0, 3)
df['VASC'] = df['VASC'].replace(1.0, 3)
df['SCC'] = df['SCC'].replace(1.0, 3)
df['SCC'].value_counts()


# In[5]:


df.drop_duplicates()
df


# In[ ]:





# In[6]:


df1=df[['MEL','BCC','AK','DF','VASC','SCC']]
df1=df1.astype(int)


# In[7]:


# Creating Label column to be used in the model


df1['LABEL']=df['MEL']+df['BCC']+df['AK']+df['DF']+df['VASC']+df['SCC']
df1['LABEL']=df1['LABEL'].astype(int)


# In[8]:


df1


# In[9]:


df1['LABEL'].value_counts()


# In[10]:


# concatinating df1 + df for images name


df2=pd.concat([df['image'],df1 ],axis=1)
df2


# In[11]:


df_3 = df2.drop(df2[df2['LABEL']==0].index).reset_index(drop=True)


# In[12]:


# counting types

df_3[['MEL','BCC','AK','DF','VASC','SCC']].value_counts()


# In[13]:


df_3[['LABEL']].value_counts()


# In[14]:


# Creating the path for image name to concatinating with the dataset

df_3['folder']="C:\\Users\\User\\Downloads\\ISIC_2019_Training_Input\\ISIC_2019_Training_Input\\"
df_3['path']=(df_3['folder']+df_3['image'])+".jpg"
df_3['path']


# In[15]:


df_3['JPG'] = df_3['path'].map(lambda x: np.asarray(Image.open(x).resize((32,32))))
df_3


# In[19]:


n_samples = 3  # number of samples for plotting
# Plotting
fig, m_axs = plt.subplots(3, n_samples, figsize = (4*n_samples, 3*3))
for n_axs, (type_name, type_rows) in zip(m_axs, 
                                         df_3.sort_values(['LABEL']).groupby('LABEL')):
    n_axs[0].set_title(type_name)
    for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=1234).iterrows()):
        c_ax.imshow(c_row['JPG'])
        c_ax.axis('off')


# In[111]:


# Shuflling and reseting index

df_3=df_3.sample(frac=1)
df_3=df_3.reset_index(drop=True)
df_3


# In[112]:


# Preparing the model 

X = np.asarray(df_3['JPG'].tolist())
X = X/255  # Scale values to 0-1. You can also used standardscaler or other scaling methods.
Y=df_3[['LABEL']]  #Assign label values to Y
#Y_cat = to_categorical(Y, num_classes=7) #Convert to categorical as this is a multiclass classification problem
#Split to training and testing
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)


# In[113]:


Y


# In[114]:


import autokeras as ak


# In[115]:


# Autokeras modeling

clf = ak.ImageClassifier(
    overwrite=True,
    max_trials=1)
# Feed the image classifier with training data.
history = clf.fit(x_train, y_train)


# In[116]:


# Model output

model = clf.export_model()
model.summary()


# In[117]:


# model evaluation

print(clf.evaluate(x_test, y_test))


# In[164]:


y=df_3

x=y['JPG'].sample(1)
image  = np.asarray(x.tolist())

predicted_y = clf.predict(image)




print('Actual type:',y['LABEL'].iloc[x.index[0]])
print('Predicted type:')
#print(x1)
#print(round(x1.max()))
#classes = np.argmax(predicted_y, axis = 1)
#if predicted_y == 0:
    #print('NO')
#if predicted_y == 1:
    #print('YES')
# print(predicted_y)
print(predicted_y)


# In[121]:


y_pred=clf.predict(x_test)


# In[125]:


y_pred = clf.predict(x_test)
# Convert predictions classes to one hot vectors 
y_pred_classes = np.argmax(y_pred, axis = 1) 
# Convert test data to one hot vectors


# In[127]:


y_true


# In[133]:


pd.DataFrame(y_test).value_counts()


# In[141]:


y_pred=(y_pred).astype(int)


# In[143]:


cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(8,8))
sns.set(font_scale=1)
sns.heatmap(cm, annot=True, linewidths=.5, ax=ax, fmt="d")


# In[146]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)


# In[147]:


accuracy


# In[182]:


from sklearn.metrics import recall_score
recall = recall_score(y_test, y_pred,average='macro')
recall

