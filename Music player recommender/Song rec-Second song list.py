#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


def get_playlist_tracks(username, playlist_id):
    results = sp.user_playlist_tracks(username,playlist_id,market="GB")
    tracks = results['items']
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])
    return tracks


# In[4]:


hits1=get_playlist_tracks("spotify", "37i9dQZF1DX0kbJZpiYdZl")


# In[49]:


def get_playlist_tracks(username, playlist_id):
    results = sp.user_playlist_tracks(username,playlist_id,market="GB")
    tracks = results['items']
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])
    return tracks


# In[6]:


list_of_audio_features=[]
for item in range(0,49):
    #print (tracks[item]["track"]["id"])
    list_of_audio_features.append(sp.audio_features(hits1[item]["track"]["id"])[0])


# In[ ]:


hits1=pd.DataFrame(list_of_audio_features)    
hits1['label']='hits'


# In[ ]:





# In[10]:


hits2=get_playlist_tracks("spotify", "6cxs2lOOJje0WDqyVE6LbX")


# In[11]:


list_of_audio_features=[]
for item in range(0,352):
    #print (tracks[item]["track"]["id"])
    list_of_audio_features.append(sp.audio_features(hits2[item]["track"]["id"])[0])


# In[12]:


hits2=pd.DataFrame(list_of_audio_features)    
hits2['label']='hits'


# In[ ]:





# In[ ]:





# In[16]:


rock=get_playlist_tracks("spotify", "7anzGe05Fe2CC3wCLnLU4f")


# In[17]:


list_of_audio_features=[]
for item in range(0,400):
    #print (tracks[item]["track"]["id"])
    list_of_audio_features.append(sp.audio_features(rock[item]["track"]["id"])[0])


# In[18]:


rock=pd.DataFrame(list_of_audio_features)    
rock['label']='rock'


# In[ ]:





# In[20]:


techno=get_playlist_tracks("spotify", "18vUeZ9BdtMRNV6gI8RnR6")


# In[21]:


list_of_audio_features=[]
for item in range(0,149):
    #print (tracks[item]["track"]["id"])
    list_of_audio_features.append(sp.audio_features(techno[item]["track"]["id"])[0])


# In[24]:


techno=pd.DataFrame(list_of_audio_features)    
techno['label']='techno'


# In[ ]:





# In[28]:


dance=get_playlist_tracks("spotify", "1ADtcUYFISue4ua9pYjo9v")


# In[29]:


list_of_audio_features=[]
for item in range(0,119):
    #print (tracks[item]["track"]["id"])
    list_of_audio_features.append(sp.audio_features(dance[item]["track"]["id"])[0])


# In[30]:


dance=pd.DataFrame(list_of_audio_features)    
dance['label']='dance'


# In[ ]:





# In[ ]:





# In[32]:


rap=get_playlist_tracks("spotify", "4riovLwMCrY3q0Cd4e0Sqp")


# In[34]:


list_of_audio_features=[]
for item in range(0,288):
    #print (tracks[item]["track"]["id"])
    list_of_audio_features.append(sp.audio_features(rap[item]["track"]["id"])[0])


# In[35]:


rap=pd.DataFrame(list_of_audio_features)    
rap['label']='rap'


# In[ ]:





# In[38]:


r_n_b=get_playlist_tracks("spotify", "37i9dQZF1DWTbCTQUrf6sj")

list_of_audio_features=[]
for item in range(0,99):
    #print (tracks[item]["track"]["id"])
    list_of_audio_features.append(sp.audio_features(r_n_b[item]["track"]["id"])[0])
    
r_n_b=pd.DataFrame(list_of_audio_features)    
r_n_b['label']='r_n_b'


# In[ ]:





# In[43]:


heavy_metal=get_playlist_tracks("spotify", "1yMlpNGEpIVUIilZlrbdS0")

list_of_audio_features=[]
for item in range(0,199):
    #print (tracks[item]["track"]["id"])
    list_of_audio_features.append(sp.audio_features(heavy_metal[item]["track"]["id"])[0])
    
heavy_metal=pd.DataFrame(list_of_audio_features)    
heavy_metal['label']='heavy_metal'


# In[ ]:





# In[44]:


popular=get_playlist_tracks("spotify", "5ABHKGoOzxkaa28ttQV9sE")

list_of_audio_features=[]
for item in range(0,99):
    #print (tracks[item]["track"]["id"])
    list_of_audio_features.append(sp.audio_features(popular[item]["track"]["id"])[0])
    
popular=pd.DataFrame(list_of_audio_features)    
popular['label']='popular'


# In[ ]:





# In[48]:


eightinighti=get_playlist_tracks("spotify", "2hlsujw9cWHaDG95eXitsm")

list_of_audio_features=[]
for item in range(0,428):
    #print (tracks[item]["track"]["id"])
    list_of_audio_features.append(sp.audio_features(eightinighti[item]["track"]["id"])[0])
    
eightinighti=pd.DataFrame(list_of_audio_features)    
eightinighti['label']='80_90'


# In[ ]:





# In[ ]:





# In[49]:


song_d2 = pd.concat([hits1,hits2,rock,techno,dance,rap,r_n_b,classical,popular,eightinighti])
song_d2


# In[51]:


song_d2.to_csv("song_df2.csv")

