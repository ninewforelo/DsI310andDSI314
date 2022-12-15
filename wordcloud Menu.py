#!/usr/bin/env python
# coding: utf-8

# # Import Library

# In[29]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import STOPWORDS
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords


# # Loading dataset

# In[30]:


w = pd.read_csv(r'C:\Users\COM01\Downloads\Woognai_Cooking_all.csv')
w


# # Worldcould

# In[21]:


pd.set_option('display.max_colwidth', None)


# In[23]:


text = " ".join(i for i in w.Name)


# In[24]:


text = ''
for row in w.Name:
    text = text + row.lower() + ''


# In[25]:


path = r"C:\Users\COM01\Downloads\THSarabunNew.ttf"


# In[33]:


wordcloud = WordCloud(font_path=path,
                      stopwords=thai_stopwords(),
                      background_color="white",
                      min_font_size=1,
                      width=1024,
                      height=768, 
                      max_words=500,
                      collocations=False,
                      regexp=r"[ก-๙a-zA-Z']+",
                      margin=2
                      ).generate(text)


# # Visualization

# In[34]:


plt.figure(figsize=(15,10))
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# In[ ]:




