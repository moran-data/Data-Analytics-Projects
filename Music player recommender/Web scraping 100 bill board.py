#!/usr/bin/env python
# coding: utf-8

# In[3]:


import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


c_id = '9a33bf98e308493c947634989589682f'
c_se ='69addd8fdd3946e0b4bd8cb4551f7948'

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=c_id, client_secret=c_se))


# In[4]:


from bs4 import BeautifulSoup
import requests
import pandas as pd


# In[5]:


url = "https://www.billboard.com/charts/hot-100/"


# In[6]:


response = requests.get(url)


# In[7]:


response.status_code


# In[8]:


response.content


# In[9]:


soup = BeautifulSoup(response.content, 'html.parser') 


# In[10]:


soup.prettify(formatter='html')


# In[11]:


title = soup.find('title')


# In[12]:


soup.find_all('div', class_='o-chart-results-list-row-container')


# In[13]:


soup.select('li>h3')[0].get_text().strip()


# In[14]:


songs=[]
for i in range(100):
    songs.append(soup.select('li>h3')[i].get_text().strip())
print(songs)


# In[15]:


artist=[]
for i in range(100):
    artist.append(soup.select('li>h3')[i].find_next('span').get_text().strip())
print(artist)


# In[16]:


df=pd.DataFrame(list(zip(artist, songs)),
      columns=['Artist','Songs'])
df


# In[17]:


df['Artist'] = df['Artist'].astype(str).str.lower()
df['Songs'] = df['Songs'].astype(str).str.lower()
import re
for i in df['Songs']:
    re.sub('[-!@#$]', ' ',i)
df['Songs']   


# In[18]:


pip install autocorrect


# In[19]:


df['Artist'] = df['Artist'].astype(str).str.lower()
df['Songs'] = df['Songs'].astype(str).str.lower()
import re

df['Songs'] = [re.sub('[-!@#$]', ' ',i) for i in df['Songs']]
df['Songs']   


# In[212]:


from difflib import SequenceMatcher
a='Taylor'
b=df['Artist']
SequenceMatcherer(None, a, b).ratio()


# In[221]:


pip install autocorrect


# In[263]:


from autocorrect import Speller

name = 'taglor'
spell = Speller(lang='en')
corr_input = spell(name)
corr_input


# In[267]:


import re
line = re.sub('[-!@#$]', ' ', 'taylor-swift')
line


# In[203]:


searchfor = ['swift']
s=df[df['Artist'].str.contains('|'.join(searchfor))]
s


# In[134]:


searchfor = ['taylor']
z=df[df['Artist'].str.contains('|'.join(searchfor))]
z.Artist.unique()


# In[127]:


import re
matches = ['taylor']
safe_matches = [df['Artist'] for m in matches]
safe_matches


# In[123]:


for i in df['Artist']:
    if df['Artist'].str.contains(x, regex=True):
    
    

    
    


# In[41]:


import re
from autocorrect import Speller
spell = Speller(lang='en')

x=re.sub('[-!@#$]', ' ',spell(str(input('Please enter an Artist name: ')))).lower()

if df['Artist'].eq(x).any() == True:
    y=re.sub('[-!@#$]', ' ',spell(str(input('Please enter song name: ')))).lower()
    if df['Songs'].eq(y).any() == True:  
        print('the song is in the top 100')
    elif df['Songs'].str.contains(y, regex=True).any():
        searchfor = [y]
        s=df[df['Songs'].str.contains('|'.join(searchfor))]
        print('Did you mean:' + str(s.Songs.unique())+'?' + '\n' + "If not, choose another Song")
    else:
        print('Did not find any match,Try again')
elif df['Artist'].str.contains(x, regex=True).any():
    searchfor = [x]
    z=df[df['Artist'].str.contains('|'.join(searchfor))]
    #print('Did you mean:' + str(z.Artist.unique())+'?' + '\n' + "If not, choose another Artist")
    a1 = input('Did you mean:' + str(z.Artist.unique())+'?'  ).lower()
    if a1 == 'yes':
        y=re.sub('[-!@#$]', ' ',spell(str(input('Please enter song name: ')))).lower()
        if df['Songs'].eq(y).any() == True:  
            print('the song is in the top 100')
        elif df['Songs'].str.contains(y, regex=True).any():
            searchfor = [y]
            s=df[df['Songs'].str.contains('|'.join(searchfor))]
        #print('Did you mean:' + str(s.Songs.unique())+'?' + '\n' + "If not, choose another Song")
            a2 = input('Did you mean:' + str(s.Songs.unique())+'?'  ).lower()
            if a2 == 'yes':
                print('the song is in the top 100, enjoy!')
            else:
                print('Did not find any match,Try again')
        
    
    else:
        print('Did not find any match,Try again')


    


    
    

