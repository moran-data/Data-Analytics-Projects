#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pickle
from sklearn import datasets # sklearn comes with some toy datasets to practise
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from matplotlib import pyplot
from sklearn.metrics import silhouette_score
import spotipy
import json
from spotipy.oauth2 import SpotifyClientCredentials


# In[2]:


df = pd.read_csv(r"C:\Users\User\Downloads\song_df.csv")


# In[3]:


df.drop_duplicates()
df


# In[4]:


df1=df[["danceability","energy","loudness","speechiness","acousticness",
    "instrumentalness","liveness","valence","tempo",]]
df1=df1.drop_duplicates()



# In[5]:


df1.reset_index(drop=True)


# In[6]:


scaler = StandardScaler()
scaler.fit(df1)
df1_scaled = scaler.transform(df1)
df1_scaled_df = pd.DataFrame(df1_scaled, columns = df1.columns)
display(df1.head())
print()
display(df1_scaled_df.head())


# In[7]:


kmeans = KMeans(n_clusters=11, random_state=1234)
kmeans.fit(df1_scaled_df)


# In[8]:


kmeans.labels_


# In[9]:


# assign a cluster to each example
labels = kmeans.predict(df1_scaled_df)
# retrieve unique clusters
clusters = np.unique(labels)
# create scatter plot for samples from each cluster
for cluster in clusters:
    # get row indexes for samples with this cluster
    row_ix = np.where(labels == cluster)
    # create scatter of these samples
    pyplot.scatter(df1_scaled_df.to_numpy()[row_ix, 1], df1_scaled_df.to_numpy()[row_ix, 3])
    # show the plot
pyplot.show()


# In[10]:


clusters = kmeans.predict(df1_scaled_df)
#clusters
pd.Series(clusters).value_counts().sort_index()


# In[11]:


kmeans2 = KMeans(n_clusters=3,
                init="k-means++",
                n_init=50,  # try with 1, 4, 8, 20, 30, 100...
                max_iter=1,
                tol=0,
                algorithm="elkan",
                random_state=1234)
kmeans2.fit(df1_scaled_df)
print(kmeans2.inertia_)


# In[12]:


K = range(2, 21)
inertia = []

for k in K:
    print("Training a K-Means model with {} clusters! ".format(k))
    print()
    kmeans = KMeans(n_clusters=k,
                    random_state=1234)
    kmeans.fit(df1_scaled_df)
    inertia.append(kmeans.inertia_)

import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure(figsize=(16,8))
plt.plot(K, inertia, 'bx-')
plt.xlabel('k')
plt.ylabel('inertia')
plt.xticks(np.arange(min(K), max(K)+1, 1.0))
plt.title('Elbow Method showing the optimal k')


# In[13]:


K = range(2, 20)
silhouette = []

for k in K:
    kmeans = KMeans(n_clusters=k,
                    random_state=1234)
    kmeans.fit(df1_scaled_df)
    
    filename = "kmeans_" + str(k) + ".pickle"
    with open(filename, "wb") as f:
        pickle.dump(kmeans,f)
    
    silhouette.append(silhouette_score(df1_scaled_df, kmeans.predict(df1_scaled_df)))


plt.figure(figsize=(16,8))
plt.plot(K, silhouette, 'bx-')
plt.xlabel('k')
plt.ylabel('silhouette score')
plt.xticks(np.arange(min(K), max(K)+1, 1.0))
plt.title('Silhouette Method showing the optimal k')


# In[14]:


df2=pd.merge(df1_scaled_df,df['id'],left_index=True,right_index=True)
df2


# In[15]:


df2["cluster"] = clusters
df2


# # Saving model

# In[16]:


import pickle

scaler = StandardScaler()
model = KMeans()

def save(model, filename = 'filename.pickle'): 
    with open(filename, "wb") as f:
        pickle.dump(model, f)


# In[17]:


save(model, filename = 'filename.pickle')


# # Test
# 

# In[18]:


from IPython.display import IFrame

#track_id = "1rfORa9iYmocEsnnZGMVC4"
track_id= 'spotify:track:3hgl7EQwTutSm6PESsB7gZ'
IFrame(src="https://open.spotify.com/embed/track/"+track_id,
       width="320",
       height="80",
       frameborder="0",
       allowtransparency="true",
       allow="encrypted-media",
      )


# In[19]:


def play_song(track_id):
    return IFrame(src="https://open.spotify.com/embed/track/"+track_id,
       width="320",
       height="80",
       frameborder="0",
       allowtransparency="true",
       allow="encrypted-media",
      )


# In[20]:


play_song


# <b> getting the track id

# In[ ]:





# In[ ]:





# In[21]:


import spotipy
import json
from spotipy.oauth2 import SpotifyClientCredentials


#Initialize SpotiPy with user credentias
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id= '9a33bf98e308493c947634989589682f',
                                                           client_secret= '69addd8fdd3946e0b4bd8cb4551f7948'))


# In[ ]:





# In[22]:


x = input('Please choose song name that you like:\n')
x1 = sp.search(x,limit=3,market="US")
print('Please select one of the options below:\n')
for item,x in zip(x1['tracks']['items'], range(1,4)):
    
    print("'{}', by the Artist: {}".format(item['name'],item["artists"][0]["name"]),
          ', Please enter -',x )
x2=int(input('Your selection: '))
if x2 == 1 or x2 == 2 or x2 == 3 :
    af=sp.audio_features(x1["tracks"]["items"][x2-1]["id"])
    af=pd.DataFrame(af)
    af1=af[["danceability","energy","loudness","speechiness","acousticness",
    "instrumentalness","liveness","valence","tempo",]]
     
    scaler.fit(af1)
    dfy_scaled = scaler.transform(af1)
    cluster = kmeans.predict(dfy_scaled)
    no=cluster[0]
    choice=df2[df2['cluster']==no]
    result=choice.sample()
    st=result['id'].values
    st1=(' '.join(st))

    print('Here is our recommendation:\n')
    
play_song(st1)
#else:
    #(print('Please choose again a song name'))

