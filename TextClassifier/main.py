#!/usr/bin/env python
# coding: utf-8

# # TEXT CLASSIFICATION USING NAIVE BAYES

# Hi, here we will be building a simple text classifier using the Naive Bayes Algorithm + the confusion matrix.
# Moreover we will also be fine-tuning it on the go for better accuracy at the end.
# ### What is Naive Bayes Algorithm ?
# This alogrithm is based on the Baye's Algorithm from the  probability theory.
# 
# Formula : $P((C|X) = P(C).P(X|C))/P(X)$
# 
# H -> Class,
# D -> Text
# 
# Naive Baye's algorithm assumes every word in independent in a particular class.
# Now, if every word is independent then :
# 
#  $P(X|C) = P(x1, x2,...,xn|C) = P(X1|C)P(X2|C)...P(Xn|C) $
# 
#  We put this assumption in the Naive Baye's Formula we get :
# 
# 
# $P((C|X) = P(C)∏(n,i=1)P(xi|C))$
# 
# Hence, to predict we use the most probable class of an input by computing the probability of each class and selecting the one with the highest probability. We have a formula
# 
# $Most Probable Class =argmaxP(C)∏(i=1,n)P(Xi|C)$
# 

# Now, let's start coding to get the idea how this works.

# for checking if we are in correct jupyter kernel or not
# import sys
# print('jupyter' in sys.modules)

# In[5]:


import os

# Install spaCy
os.system("pip install -U spacy==3.*")

# Download the English model
os.system("python -m spacy download en_core_web_sm")

# Check spaCy installation info
os.system("python -m spacy info")


# In[6]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
import time  

from sklearn import metrics
from sklearn import model_selection
from sklearn.datasets import fetch_20newsgroups
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


# In[4]:


train_corpus = fetch_20newsgroups(subset = "train")


# In[5]:


print("Training data size : {}".format(len(train_corpus.data)))


# In[6]:


train_corpus.target_names


# In[7]:


print(train_corpus.target)


# In[8]:


# first post along with the corresponding label
print(train_corpus.data[0])

print("Label no. {}".format(train_corpus.target[0]))
print("Label name : {}".format(train_corpus.target_names[0]))


# In[9]:


bins, counts = np.unique(train_corpus.target, return_counts=True)
freq_series = pd.Series(counts/len(train_corpus.data))
plt.figure(figsize=(12, 8))
ax = freq_series.plot(kind='bar')
ax.set_xticklabels(bins, rotation=0)
plt.show()


# In[10]:


train_data, val_data, train_target, val_target = train_test_split(train_corpus.data, train_corpus.target, train_size= 0.8,random_state= 1)
print("training data size : {}".format(len(train_data)))
print("validation data size : {}".format(len(val_data)))


# In[11]:


nlp = spacy.load("en_core_web_sm")


# In[12]:


nlp.pipe_names


# In[13]:


nlp = spacy.blank('en')
nlp.pipe_names


# In[15]:


def spacy_tokenizer(doc):
 return [t.text for t in nlp(doc) if(not t.is_punct and not t.is_space and t.is_alpha)]


# In[14]:


#vectorize using TfIdfVectorizer


start_time = time.time()  

sum(range(1000000))  

end_time = time.time()  

print(f"Execution time: {end_time - start_time:.5f} seconds") 

vectorizer = TfidfVectorizer()
train_feature_vec = vectorizer.fit_transform(train_data)


# In[16]:


nb_classifier = MultinomialNB()
nb_classifier.fit(train_feature_vec, train_target)
nb_classifier.get_params()


# In[17]:


#use F1 for measuring the accurancy
train_preds = nb_classifier.predict(train_feature_vec)
print("Initial F1 score {}".format(metrics.f1_score(train_target, train_preds, average = "macro")))


# In[18]:


# removing headers, footers and quotes from training set and resplit
filtered_training_corpus = fetch_20newsgroups(subset= "train", remove=('headers', 'footers', 'quotes'))
train_data, val_data, train_label, val_label = train_test_split(filtered_training_corpus.data, filtered_training_corpus.target, train_size= 0.8, random_state= 1)


# In[19]:

start_time = time.time()

train_feature_vec = vectorizer.fit_transform(train_data)
nb_classifier.fit(train_feature_vec, train_label)

end_time = time.time()

print(f"Time : {end_time - start_time:.5f} seconds")

# In[20]:


train_preds = nb_classifier.predict(train_feature_vec)
print("F1 score of filtered training set : {}".format(metrics.f1_score(train_label, train_preds, average="macro")))


# In[21]:


# checking how the classifier performs on the validation set
# first vectorize the validation data
start_time = time.time()  

sum(range(1000000))  

end_time = time.time()  

print(f"Execution time: {end_time - start_time:.5f} seconds") 
val_feature_vec = vectorizer.transform(val_data)


# In[22]:


val_pred = nb_classifier.predict(val_feature_vec)
print("F1 score of filtered validation set : {}".format(metrics.f1_score(val_label,val_pred, average="macro")))


# # Confusion Matrix
# 

# In[24]:


# Set the size of the plot.
fig, ax = plt.subplots(figsize=(15, 15))

# Create the confusion matrix.
disp = ConfusionMatrixDisplay.from_estimator(nb_classifier, val_feature_vec, val_label, normalize='true', display_labels=filtered_training_corpus.target_names, xticks_rotation='vertical', ax=ax)


# In[26]:


# looking at the precision and recall
print(metrics.classification_report(val_label, val_pred, target_names = filtered_training_corpus.target_names))


# In[27]:


print("Training data size : {}".format(len(train_data)))
print("No. of training features : {}".format(len(train_feature_vec[0].toarray().flatten())))


# In[33]:


nlp = spacy.load("en_core_web_sm")


# In[34]:


not_needed = ['ner', 'parser']

# remove the stop words and add lemma instead of token text
def spacy_tokenizer(doc):
 with nlp.disable_pipes(*not_needed):
    return [ t.lemma_ for t in nlp(doc) if(not t.is_punct and not t.is_space and not t.is_stop and t.is_alpha) ]


# In[35]:


# again revectorize
start_time = time.time()  

sum(range(1000000))  

end_time = time.time()  

print(f"Execution time: {end_time - start_time:.5f} seconds") 
vectorizer = TfidfVectorizer(tokenizer=spacy_tokenizer)
train_feature_vec = vectorizer.fit_transform(train_data)


# In[36]:


print("number of features now : {}".format(len(train_feature_vec[0].toarray().flatten())))
#hence the no. of unnecessary features have reduced


# In[38]:


nb_classifier.fit(train_feature_vec,train_label)
train_preds = nb_classifier.predict(train_feature_vec)
print("training F1 score with fewer features : {}".format(metrics.f1_score(train_label, train_preds,average= "macro")))


# In[40]:


#checking classifier performance on validation set
start_time = time.time()  

sum(range(1000000))  

end_time = time.time()  

print(f"Execution time: {end_time - start_time:.5f} seconds") 
val_feature_vec = vectorizer.transform(val_data)


# In[42]:


val_pred = nb_classifier.predict(val_feature_vec)
print("Validation score with fewer features : {}".format(metrics.f1_score(val_label, val_pred, average="macro")))


# In[43]:


fig, ax = plt.subplots(figsize=(15, 15))
disp = ConfusionMatrixDisplay.from_estimator(nb_classifier, val_feature_vec, val_label, normalize='true', display_labels=filtered_training_corpus.target_names, xticks_rotation='vertical', ax=ax)


# In[45]:


print(metrics.classification_report(val_label, val_pred, target_names=filtered_training_corpus.target_names))


# In[47]:


# alpha values
params = {'alpha': [0.01, 0.1, 0.5, 1.0, 10.0,],}

# Instantiate the search with the model we want to try and fit it on the training data.
multinomial_nb_grid = model_selection.GridSearchCV(MultinomialNB(), param_grid=params, scoring='f1_macro', n_jobs=-1, cv=5, verbose=5)
multinomial_nb_grid.fit(train_feature_vec, train_label)


# In[48]:


print('Best parameter value(s): {}'.format(multinomial_nb_grid.best_params_))


# In[53]:


best_nb_classifier = multinomial_nb_grid.best_estimator_
val_pred = best_nb_classifier.predict(val_feature_vec)
print('Validation F1 score with fewer features: {}'.format(metrics.f1_score(val_label, val_pred, average='macro')))


# In[54]:


fig, ax = plt.subplots(figsize=(15, 15))
disp = ConfusionMatrixDisplay.from_estimator(best_nb_classifier, val_feature_vec, val_label, normalize='true', display_labels=filtered_training_corpus.target_names, xticks_rotation='vertical', ax=ax)


# In[56]:


print(metrics.classification_report(val_label, val_pred, target_names=filtered_training_corpus.target_names))


# # Creating Our Final Naive Baye's Classifier

# In[57]:


text_classifier = Pipeline([
    ('vectorizer', TfidfVectorizer(tokenizer=spacy_tokenizer)),
    ('classifier', MultinomialNB(alpha = 0.01))
])


# In[58]:

start_time = time.time()
text_classifier.fit(filtered_training_corpus.data, filtered_training_corpus.target)
end_time = time.time()

print(f"Execution time : {end_time - start_time:.5f} seconds")


# In[59]:


filtered_test_corpus = fetch_20newsgroups(subset="test", remove=('headers', 'footers', 'quotes'))


# In[60]:


test_pred = text_classifier.predict(filtered_test_corpus.data)


# In[62]:


start_time = time.time()

# Create the plot
fig, ax = plt.subplots(figsize=(15, 15))
ConfusionMatrixDisplay.from_predictions(
    filtered_test_corpus.target, 
    test_pred, 
    normalize='true', 
    display_labels=filtered_test_corpus.target_names, 
    xticks_rotation='vertical', 
    ax=ax
)
plt.show()

# End timing
end_time = time.time()

# Print execution time
print(f"Execution time: {end_time - start_time:.5f} seconds")


# In[64]:


print(metrics.classification_report(filtered_test_corpus.target, test_pred, target_names=filtered_test_corpus.target_names))


# In[72]:


def classify_text(clf, doc, labels=None):
  probas = clf.predict_proba([doc]).flatten()
  max_proba_idx = np.argmax(probas)

  if labels:
    most_proba_class = labels[max_proba_idx]
  else:
    most_proba_class = max_proba_idx

  return (most_proba_class, probas[max_proba_idx])


# In[73]:


# Post from r/medicine.
s = "Hello everyone so am doing my thesis on Ischemic heart disease have been using online articles and textbooks mostly Harrisons internal med. could u recommended me some source specifically books where i can get more about in depth knowledge on IHD."
classify_text(text_classifier, s, filtered_test_corpus.target_names)


# In[74]:


# Post from r/space.
s = "First evidence that water can be created on the lunar surface by Earth's magnetosphere. Particles from Earth can seed the moon with water, implying that other planets could also contribute water to their satellites."
classify_text(text_classifier, s, filtered_test_corpus.target_names)

