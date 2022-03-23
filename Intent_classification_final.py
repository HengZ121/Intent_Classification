#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/Dark-Sied/Intent_Classification/blob/master/Intent_classification_final.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[2]:


import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
import nltk
import re
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Dropout
from keras.callbacks import ModelCheckpoint


# In[ ]:


def load_dataset(filename):
  df = pd.read_csv(filename, encoding = "latin1", names = ["Sentence", "Intent"])
  print(df.head())
  intent = df["Intent"]
  unique_intent = list(set(intent))
  sentences = list(df["Sentence"])
  
  return (intent, unique_intent, sentences)
  


# In[5]:


intent, unique_intent, sentences = load_dataset("Dataset.csv")


# In[ ]:


print(sentences[:5])


# In[ ]:


nltk.download("stopwords")
nltk.download("punkt")


# In[ ]:


#define stemmer
stemmer = LancasterStemmer()


# In[ ]:


def cleaning(sentences):
  words = []
  for s in sentences:
    clean = re.sub(r'[^ a-z A-Z 0-9]', " ", s)
    w = word_tokenize(clean)
    #stemming
    words.append([i.lower() for i in w])
    
  return words  


# In[ ]:


cleaned_words = cleaning(sentences)
print(len(cleaned_words))
print(cleaned_words[:2])  
  


# In[ ]:


def create_tokenizer(words, filters = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'):
  token = Tokenizer(filters = filters)
  token.fit_on_texts(words)
  return token


# In[ ]:


def max_length(words):
  return(len(max(words, key = len)))
  


# In[ ]:


word_tokenizer = create_tokenizer(cleaned_words)
vocab_size = len(word_tokenizer.word_index) + 1
max_length = max_length(cleaned_words)

print("Vocab Size = %d and Maximum length = %d" % (vocab_size, max_length))


# In[ ]:


def encoding_doc(token, words):
  return(token.texts_to_sequences(words))


# In[ ]:


encoded_doc = encoding_doc(word_tokenizer, cleaned_words)


# In[ ]:


def padding_doc(encoded_doc, max_length):
  return(pad_sequences(encoded_doc, maxlen = max_length, padding = "post"))


# In[ ]:


padded_doc = padding_doc(encoded_doc, max_length)


# In[ ]:


padded_doc[:5]


# In[ ]:


print("Shape of padded docs = ",padded_doc.shape)


# In[ ]:


#tokenizer with filter changed
output_tokenizer = create_tokenizer(unique_intent, filters = '!"#$%&()*+,-/:;<=>?@[\]^`{|}~')


# In[ ]:


output_tokenizer.word_index


# In[ ]:


encoded_output = encoding_doc(output_tokenizer, intent)


# In[ ]:


encoded_output = np.array(encoded_output).reshape(len(encoded_output), 1)


# In[ ]:


encoded_output.shape


# In[ ]:


def one_hot(encode):
  o = OneHotEncoder(sparse = False)
  return(o.fit_transform(encode))


# In[ ]:


output_one_hot = one_hot(encoded_output)


# In[ ]:


output_one_hot.shape


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


train_X, val_X, train_Y, val_Y = train_test_split(padded_doc, output_one_hot, shuffle = True, test_size = 0.2)


# In[ ]:


print("Shape of train_X = %s and train_Y = %s" % (train_X.shape, train_Y.shape))
print("Shape of val_X = %s and val_Y = %s" % (val_X.shape, val_Y.shape))


# In[ ]:


def create_model(vocab_size, max_length):
  model = Sequential()
  model.add(Embedding(vocab_size, 128, input_length = max_length, trainable = False))
  model.add(Bidirectional(LSTM(128)))
#   model.add(LSTM(128))
  model.add(Dense(32, activation = "relu"))
  model.add(Dropout(0.5))
  model.add(Dense(21, activation = "softmax"))
  
  return model


# In[ ]:


model = create_model(vocab_size, max_length)

model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
model.summary()


# In[ ]:


filename = 'model.h5'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# hist = model.fit(train_X, train_Y, epochs = 100, batch_size = 32, validation_data = (val_X, val_Y), callbacks = [checkpoint])


# In[ ]:


model = load_model("model.h5")


# In[ ]:


def predictions(text):
  clean = re.sub(r'[^ a-z A-Z 0-9]', " ", text)
  test_word = word_tokenize(clean)
  test_word = [w.lower() for w in test_word]
  test_ls = word_tokenizer.texts_to_sequences(test_word)
  print(test_word)
  #Check for unknown words
  if [] in test_ls:
    test_ls = list(filter(None, test_ls))
    
  test_ls = np.array(test_ls).reshape(1, len(test_ls))
 
  x = padding_doc(test_ls, max_length)
  
  pred = model.predict(x)
  
  
  return pred


  


# In[ ]:


def get_final_output(pred, classes):
  predictions = pred[0]
 
  classes = np.array(classes)
  ids = np.argsort(-predictions)
  classes = classes[ids]
  predictions = -np.sort(-predictions)
 
  for i in range(pred.shape[1]):
    print("%s has confidence = %s" % (classes[i], (predictions[i])))


# In[ ]:


text = "Can you help me?"
pred = predictions(text)
get_final_output(pred, unique_intent)


# In[ ]:




