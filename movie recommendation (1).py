#!/usr/bin/env python
# coding: utf-8

# # Author: Sadaf Shaikh 

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


movies= pd.read_csv('tmdb_5000_movies.csv')
credits= pd.read_csv('tmdb_5000_credits.csv')


# In[3]:


movies.head(1)


# In[4]:


credits.head()


# In[5]:


movies.columns


# In[6]:


movies = movies.merge(credits,on='title')


# In[7]:


movies.head()


# In[8]:


movies= movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]


# In[9]:


movies.head()


# In[10]:


movies.isnull().sum()


# In[11]:


movies.dropna(inplace=True)


# In[12]:


movies.isnull().sum()


# In[13]:


movies.duplicated().sum()


# In[14]:


movies.head(1)


# In[15]:


movies.iloc[0].genres


# In[16]:


#we have got a string of list


# In[17]:


import ast
def convert(obj):
    l=[]
    for i in ast.literal_eval(obj):
        l.append(i['name'])
    return l


# In[18]:


movies['genres']= movies['genres'].apply(convert)
movies['keywords']= movies['keywords'].apply(convert)


# In[19]:


movies.head(1)


# In[20]:


#extracting only 3 actors

def convert3(obj):
    l=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter!=3:
            l.append(i['name'])
            counter +=1
            
        else:
            break
            
    return l

movies['cast'] = movies['cast'].apply(convert3)
            


# In[21]:


#extracting name of director from crew
def fetch_director(obj):
    l=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            l.append(i['name'])
            break
    return l


# In[22]:


movies['crew']= movies['crew'].apply(fetch_director)


# In[23]:


movies.head()


# In[24]:


movies['overview'][0]


# In[25]:


movies['overview']= movies['overview'].apply(lambda x : x.split())


# In[26]:


movies.head()


# In[27]:


#removing space between characters

movies['genres']= movies['genres'].apply(lambda x : [i.replace(" ", "") for i in x])
movies['keywords']= movies['keywords'].apply(lambda x : [i.replace(" ", "") for i in x])
movies['cast']= movies['cast'].apply(lambda x : [i.replace(" ", "") for i in x])
movies['crew']= movies['crew'].apply(lambda x : [i.replace(" ", "") for i in x])


# In[28]:


movies.head(1)


# In[29]:


movies['tags']= movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[30]:


movies.head()


# In[31]:


new_df= movies[['movie_id' , 'title', 'tags']]


# In[32]:


new_df.head()


# In[33]:


new_df['tags'][0]


# In[34]:


new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df.head()


# In[35]:


new_df['tags']= new_df['tags'].apply(lambda x: x.lower())


# In[36]:


new_df.head()


# In[37]:


from sklearn.feature_extraction.text import CountVectorizer

cv= CountVectorizer(max_features=5000 , stop_words='english')


# In[38]:


vectors=cv.fit_transform(new_df['tags']).toarray()


# In[39]:


vectors.shape


# In[40]:


cv.get_feature_names_out()


# In[41]:


from sklearn.metrics.pairwise import cosine_similarity


# In[42]:


similarity = cosine_similarity(vectors)


# In[43]:


similarity


# In[44]:


similarity.shape


# In[45]:


new_df[new_df['title'] == 'The Lego Movie'].index[0]


# In[46]:


def recommend(movie):
    index = new_df[new_df['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        print(new_df.iloc[i[0]].title)


# In[47]:


recommend('Gandhi')


# In[48]:


import pickle


# In[49]:


pickle.dump(new_df,open('movie_list.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:





# In[ ]:




