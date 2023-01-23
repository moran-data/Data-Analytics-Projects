#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


df = pd.read_csv(r"C:\Users\User\Downloads\ISIC_2019_Training_GroundTruth.csv")


# In[3]:


df.drop_duplicates()
df


# Types of Benign and malignant cancer 
# We would gorup the types to 2 Categories - 
# 
# Malignant-
#     1.Melanoma
#     2.Basal cell carcinoma
#     3.Actinic keratosis -  The patches are not usually serious / cancer
#     4.Dermatofibroma -  Dermatofibromas are referred to as benign fibrous 
#     5.Vascular lesion - may be benign (not cancer) or malignant (cancer) 
#     6.Squamous cell carcinoma - In general, the squamous cell carcinoma survival rate is very high
# 
# Benign -    
#     1.Melanocytic nevus
#     2.Benign keratosis (solar lentigo / seborrheic keratosis / lichen planus-like keratosis)

# In[4]:


df[['MEL','NV','BCC','AK','BKL','DF','VASC','SCC','UNK']].value_counts()


# In[5]:


# Creating Label column to be used with the model

df['POSITIVE']=df['MEL']+df['BCC']+df['AK']+df['DF']+df['VASC']+df['SCC']+df['UNK']
df['NEGATIVE']=df['NV']+df['BKL']
df['LABEL']=df['POSITIVE']


# In[6]:


df


# In[7]:


df['LABEL'].value_counts()


# # BALANCING

# In[8]:


# Downsizing the benign type 

from sklearn.utils import resample

df_0 = df[df['LABEL'] == 0]
df_1 = df[df['LABEL'] == 1]

df_0_balanced = df_0.sample(5000) 
df_1_balanced = df_1.sample(5000) 



#Combined back to a single dataframe
df_balanced = pd.concat([df_0_balanced, df_1_balanced])


# In[9]:


df_balanced['LABEL'].value_counts()


# In[10]:


df_balanced['LABEL']=df_balanced['LABEL'].astype(int)


# In[11]:


df_balanced=df_balanced.sample(frac=1)
df_balanced=df_balanced.reset_index(drop=True)
df_balanced


# In[12]:


# Creating the path column for image name to concatinating with the dataset


df_balanced['folder']="C:\\Users\\User\\Downloads\\ISIC_2019_Training_Input\\ISIC_2019_Training_Input\\"
df_balanced['path']=(df_balanced['folder']+df_balanced['image'])+".jpg"
df_balanced['path']


# In[13]:


# concatinating the image as numpy 3d array

df_balanced['JPG'] = df_balanced['path'].map(lambda x: np.asarray(Image.open(x).resize((200,200))))
df_balanced


# In[14]:


# preparing the model 

X = np.asarray(df_balanced['JPG'].tolist())
X = X/255  
Y=df_balanced['LABEL']  
#Y_cat = to_categorical(Y, num_classes=7) 
#Split to training and testing
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)


# In[15]:


Y


# In[16]:


import autokeras as ak


# In[ ]:


# Autokeras model

clf = ak.ImageClassifier(
    overwrite=True,
    max_trials=1)
# Feed the image classifier with training data.
history = clf.fit(x_train, y_train)


# In[ ]:


model = clf.export_model()
model.summary()


# In[64]:


# Evaluating the model

print(clf.evaluate(x_test, y_test))


# In[67]:


y=df_balanced

x=y['JPG'].sample(1)
image  = np.asarray(x.tolist())

predicted_y = clf.predict(image)




print('Actual type:',y['LABEL'].iloc[x.index[0]])
print('Predicted type:')
#print(x1)
#print(round(x1.max()))
classes = np.argmax(predicted_y, axis = 1)
if predicted_y == 0:
    print('NO')
if predicted_y == 1:
    print('YES')
# print(predicted_y)
plt.imshow(np.squeeze(image))


# In[66]:


x.index[0]


# In[22]:


df_balanced.iloc[4113]


# In[299]:


print(predicted_y)


# In[298]:


y_pred = clf.predict(x_test)
y_pred =(y_pred>0.5)
list(y_pred)

cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[ ]:


#  Trying the Sequential model


# In[371]:


X = np.asarray(df_balanced['JPG'].tolist())
X = X/255  
Y=df_balanced['LABEL']  
#Y_cat = to_categorical(Y, num_classes=2) 
#Split to training and testing
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)


# In[372]:


Y


# In[374]:


#  Sequential model with 10 hidden layers and labels 0,1 as one column

num_classes = 2

model = Sequential()
model.add(Conv2D(256, (3, 3), activation="relu", 
                 bias_initializer="zeros", 
                 input_shape=(64, 64, 3)))
#model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))  
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3),activation='relu', 
                 bias_initializer="zeros"))
#model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))  
model.add(Dropout(0.3))

model.add(Conv2D(64, (3, 3),activation='relu',
                bias_initializer="zeros"))
#model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))  
model.add(Dropout(0.3))
model.add(Flatten())

model.add(Dense(64))
model.add(Dense(1, activation='softmax'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['acc'])


# In[376]:


batch_size = 10
epochs = 15

history = model.fit(
    x_train, y_train,
    epochs=epochs,
    batch_size = batch_size,
    validation_data=(x_test, y_test),
    verbose=2)

score = model.evaluate(x_test, y_test)
print('Test accuracy:', score[1])


# In[352]:


y=df_balanced

x=y['JPG'].sample(1)
image  = np.asarray(x.tolist())

predicted_y = model.predict(image)




print('Actual type:',y['LABEL'].iloc[x.index[0]])
print('Predicted type:')
#print(x1)
#print(round(x1.max()))
#classes = np.argmax(predicted_y, axis = 1)
#if predicted_y == 0:
    #print('NO')
#if predicted_y == 1:
    #print('YES')
print(predicted_y)


# In[365]:


y_predict=model.predict(x_test)


# In[366]:


y_predict


# In[ ]:


# Second sequential model but labeling the types to numpy array


# In[384]:


X = np.asarray(df_balanced['JPG'].tolist())
X = X/255  
Y=df_balanced['LABEL']  
Y_cat = to_categorical(Y, num_classes=2) 
#Split to training and testing
x_train, x_test, y_train, y_test = train_test_split(X, Y_cat, test_size=0.25, random_state=42)


# In[386]:


Y_cat


# In[387]:


num_classes = 2

model = Sequential()
model.add(Conv2D(256, (3, 3), activation="relu", 
                 bias_initializer="zeros", 
                 input_shape=(64, 64, 3)))
#model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))  
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3),activation='relu', 
                 bias_initializer="zeros"))
#model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))  
model.add(Dropout(0.3))

model.add(Conv2D(64, (3, 3),activation='relu',
                bias_initializer="zeros"))
#model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))  
model.add(Dropout(0.3))
model.add(Flatten())

model.add(Dense(64))
model.add(Dense(2, activation='softmax'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['acc'])


# In[388]:


batch_size = 10
epochs = 15

history = model.fit(
    x_train, y_train,
    epochs=epochs,
    batch_size = batch_size,
    validation_data=(x_test, y_test),
    verbose=2)

score = model.evaluate(x_test, y_test)
print('Test accuracy:', score[1])


# In[ ]:


# Evaluating the model - Better results


# In[389]:


y_predict=model.predict(x_test)


# In[390]:


y_predict


# In[424]:


pd_pred=pd.DataFrame(y_predict)

pd_pred=pd_pred[1].to_numpy()

pd_pred


for i in y_pred2:
    


# In[417]:


pd_y_predict=[]
for i in pd_pred:
    if i > 0.5:
        pd_y_predict.append(1)
    if i < 0.5:
        pd_y_predict.append(0)

    


# In[414]:


pd_y_predict


# In[418]:


y_test


# In[419]:


pd_test=pd.DataFrame(y_test)
pd_test=pd_test[1].to_numpy()
pd_test


# In[421]:


from sklearn.metrics import recall_score
recall = recall_score(pd_test, pd_y_predict)
recall

