#!/usr/bin/env python
# coding: utf-8

# In[10]:


import config
import pandas as pd


# In[2]:


import spotipy
import json
from spotipy.oauth2 import SpotifyClientCredentials


#Initialize SpotiPy with user credentias
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id= config.Client_ID,
                                                           client_secret= config.Client_Secret))


# In[3]:


playlist = sp.user_playlist_tracks("spotify", "5fo41o54DPTvdPO2uMTDH1",market="GB")


# In[4]:


playlist["items"][0]


# In[7]:


tracks


# In[58]:


list_of_audio_features=[]
for item in range(0,10):
    #print (tracks[item]["track"]["id"])
    list_of_audio_features.append(sp.audio_features(tracks[item]["track"]["id"])[0])
    
list_of_audio_features


# In[11]:


df=pd.DataFrame(list_of_audio_features)    
df=df[["danceability","energy","loudness","speechiness","acousticness",
    "instrumentalness","liveness","valence","tempo","id","duration_ms"]]

df


# In[12]:


def get_playlist_tracks(username, playlist_id):
    results = sp.user_playlist_tracks(username,playlist_id,market="GB")
    tracks = results['items']
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])
    return tracks


# In[13]:


tracks=get_playlist_tracks("spotify", "5Q7JPVmkNBFbPF5QVJFLds")


# In[14]:


tracks


# In[33]:


list_of_audio_features=[]
for item in range(0,200):
    #print (tracks[item]["track"]["id"])
    list_of_audio_features.append(sp.audio_features(tracks[item]["track"]["id"])[0])


# In[35]:


df1=pd.DataFrame(list_of_audio_features)    
df1=df1[["danceability","energy","loudness","speechiness","acousticness",
    "instrumentalness","liveness","valence","tempo","id","duration_ms"]]


# In[44]:


tracks2=get_playlist_tracks("spotify", "5oZjXwvrbAJ29Beza3h1bn")


# In[49]:


def get_playlist_tracks(username, playlist_id):
    results = sp.user_playlist_tracks(username,playlist_id,market="GB")
    tracks = results['items']
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])
    return tracks


# In[55]:


tracks3 = get_playlist_tracks("spotify", '5fo41o54DPTvdPO2uMTDH1')


# In[56]:


list_of_audio_features=[]
for item in range(0,150):
    #print (tracks[item]["track"]["id"])
    list_of_audio_features.append(sp.audio_features(tracks3[item]["track"]["id"])[0])


# In[57]:


df3=pd.DataFrame(list_of_audio_features)    
df3


# In[59]:


tracks4 = get_playlist_tracks("spotify", '0I77w4YsrOdarNjkTfRFTj')


# In[74]:


list_of_audio_features=[]
for item in range(0,900):
    list_of_audio_features.append(sp.audio_features(tracks4[item]["track"]["id"])[0])


# In[75]:


df4=pd.DataFrame(list_of_audio_features)


# In[84]:


tracks5 = get_playlist_tracks("spotify", '4cRLEKDSH5Y64Xlj25kORM')


# In[85]:


list_of_audio_features=[]
for item in range(0,400):
    list_of_audio_features.append(sp.audio_features(tracks5[item]["track"]["id"])[0])


# In[86]:


df5=pd.DataFrame(list_of_audio_features)


# In[89]:


tracks6 = get_playlist_tracks("spotify", '72LA3OR3WCoXu6ZC7opyz9')


# In[93]:


list_of_audio_features=[]
for item in range(0,1500):
    list_of_audio_features.append(sp.audio_features(tracks6[item]["track"]["id"])[0])


# In[94]:


df6=pd.DataFrame(list_of_audio_features)


# In[ ]:





# In[96]:


list_of_audio_features=[]
for item in range(1501,4000):
    list_of_audio_features.append(sp.audio_features(tracks6[item]["track"]["id"])[0])


# In[97]:


df7=pd.DataFrame(list_of_audio_features)


# In[ ]:


list_of_audio_features=[]
for item in range(4001,6000):
    list_of_audio_features.append(sp.audio_features(tracks6[item]["track"]["id"])[0])


# In[101]:


df8=pd.DataFrame(list_of_audio_features)


# In[ ]:





# In[106]:


list_of_audio_features=[]
for item in range(7001,9500):
    list_of_audio_features.append(sp.audio_features(tracks6[item]["track"]["id"])[0])


# In[107]:


df9=pd.DataFrame(list_of_audio_features)


# In[ ]:





# In[237]:


tracks32 = get_playlist_tracks("spotify", '36LIjrOb54q0SeSqLvH3SE')


# In[238]:


list_of_audio_features=[]
for item in range(0,410):
    list_of_audio_features.append(sp.audio_features(tracks32[item]["track"]["id"])[0])


# In[239]:


df32=pd.DataFrame(list_of_audio_features)


# In[ ]:





# In[240]:


song_d = pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15,df16,df18,df19,df20,df21,df22,df23,df24,df25,df26,df27,df28,df29,df30,df31,df32])
song_d


# In[245]:


song_d.to_csv("song_d.csv")


# In[241]:


song_d.drop_duplicates()


# In[ ]:




