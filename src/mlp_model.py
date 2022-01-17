"""#### Importing libaries"""

import gensim
import gensim.downloader
from gensim.models import KeyedVectors
import pandas
import string
import pandas as pd
import collections
import itertools
import numpy as np
import pickle 
import nltk
import sklearn
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
import random

nltk.download('brown')
nltk.download('universal_tagset')
np.random.seed(5)

import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical

# Tensorflow GPU check
tf.test.is_gpu_available(
    cuda_only=False, min_cuda_compute_capability=None
)

"""### Loading and shuffling data"""

data = ""
with open('../data/Brown_tagged_train.txt', 'r') as file:
    data = file.readlines()
    data = [line.rstrip() for line in data]

file.close()

# randomly shuffle data
np.random.shuffle(data)
print("Data size =", len(data))
data[:5]   # For checking

"""### Pre-processing data"""

# parsing sentence tokens in the form (word, tag)
def custom_split(token):
    idx = token.rfind('/')
    word = token[:idx].lower()
    tag = token[idx+1:]
    return word, tag

pos_data = [i.strip().split() for i in data]
# print(pos_data[0])

tagged_data = []
for sent in pos_data:
    tagged_sent = list(map(custom_split, sent))
    tagged_data.append(tagged_sent)

tagged_data[:1]

tag_set = set([ tag for sentence in tagged_data for _ , tag in sentence])
tag_set = list(tag_set)
n_classes = len(tag_set)
print(f'No. of tags = {len(tag_set)}')
print(f'Tags = {tag_set}')

# Label Encoding for categorical attribute
tag_label_dict = {tag_set[i]:i for i in range(len(tag_set))}
tag_label_dict

"""### Adding word embeddings

"""

def add_embeddings(data, embed='glove', dim=300):
    
    # 300-dim vectors
    if embed == 'word2vec':
        print("Adding Word2vec embeddings")
        emb_load = 'word2vec-google-news-300.kv'
    elif embed == 'glove':
        print("Adding GloVe embeddings")
        emb_load = 'glove-wiki-gigaword-300.kv'
    else:
        print("Invalid word embedding model!")
        return None

    # model = gensim.downloader.load(emb_load)
    emb_model = KeyedVectors.load('../embeddings/'+emb_load)

    # np.random.shuffle(data)
    emb_data_X = []
    emb_data_y = []

    for sent in data:
        for word, tag in sent:
            if word not in emb_model.vocab:
                # random initialisation
                word_emb = np.random.rand(dim)
            else:
                word_emb = emb_model[word]     
            emb_data_X.append(word_emb)
            emb_data_y.append(tag_label_dict[tag])

    emb_data_X = np.array(emb_data_X, dtype=float)
    emb_data_y = np.array(emb_data_y, dtype=int)
    return emb_data_X, emb_data_y

"""### MLP Model for Classification"""

# Implement tensorflow model for classification

# define classification model
def classification_model(input_dim, num_classes):
    # create model
    model = Sequential()
    model.add(Dense(200, activation='relu', input_shape=(input_dim,)))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    
    # compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

"""### 1. Glove embeddings

Adding word embeddings
"""

X, y = add_embeddings(tagged_data, embed='glove')
y = to_categorical(y)
print(X.shape, y.shape)

"""K-Fold Cross Validation (K=3)"""

kf = KFold(n_splits=3)
print(kf)
kf.get_n_splits(X)

mlp_model = None
history = None
y_pred = None
y_test = None
fold = 0

test_acc = []
test_prec = []
test_rec = []
test_f1 = []

for train_index, test_index in kf.split(X):
    fold += 1
    print(f'\nFOLD - {fold}')
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # Call model
    mlp_model = classification_model(300, n_classes)
    history = mlp_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, verbose=2)

    # Predictions
    y_pred_cat = mlp_model.predict(X_test)
    y_pred = np.argmax(y_pred_cat, axis=1)

    # Accuracy
    result = mlp_model.evaluate(X_test, y_test, verbose=0)
    model_accuracy = result[1]
    y_test = np.argmax(y_test, axis=1)
    model_precision = metrics.precision_score(y_test, y_pred, average='weighted')
    model_recall = metrics.recall_score(y_test, y_pred, average='weighted')
    model_f1 = metrics.f1_score(y_test, y_pred, average='weighted')

    test_acc.append(model_accuracy)
    test_prec.append(model_precision)
    test_rec.append(model_recall)
    test_f1.append(model_f1)
    print(f">>> Fold-{fold} accuracy of MLP model = {(100*model_accuracy):.4f} %\n")

print('RESULTS: K-Fold Cross Validation')
print(f'Average accuracy after {3}-fold cross validation = {np.mean(test_acc):.4f}')
print(f'Average precision after {3}-fold cross validation = {np.mean(test_prec):.4f}')
print(f'Average recall after {3}-fold cross validation = {np.mean(test_rec):.4f}')
print(f'Average f1-score after {3}-fold cross validation = {np.mean(test_f1):.4f}')

"""Classification Report"""

print(classification_report(y_test, y_pred, target_names=tag_set))
# print(classification_report(y_test, y_pred))

"""Confusion Matrix"""

plt.figure(figsize=(12,8))
cf = confusion_matrix(y_test, y_pred)
cf_plot = sns.heatmap(cf, annot=True, cmap='GnBu', xticklabels=tag_set, yticklabels=tag_set)
plt.title("CONFUSION MATRIX")
plt.xlabel('PREDICTED Labels')
plt.ylabel('TRUE Labels')
plt.show()

"""### 2. Word2vec embeddings

Adding word embeddings
"""

X, y = add_embeddings(tagged_data, embed='word2vec')
y = to_categorical(y)
print(X.shape, y.shape)

"""K-Fold Cross Validation (K=3)"""

kf = KFold(n_splits=3)
print(kf)
kf.get_n_splits(X)

mlp_model = None
history = None
y_pred = None
y_test = None
fold = 0

test_acc = []
test_prec = []
test_rec = []
test_f1 = []

for train_index, test_index in kf.split(X):
    fold += 1
    print(f'\nFOLD - {fold}')
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # Call model
    mlp_model = classification_model(300, n_classes)
    history = mlp_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, verbose=2)

    # Predictions
    y_pred_cat = mlp_model.predict(X_test)
    y_pred = np.argmax(y_pred_cat, axis=1)

    # Accuracy
    result = mlp_model.evaluate(X_test, y_test, verbose=0)
    model_accuracy = result[1]
    y_test = np.argmax(y_test, axis=1)
    model_precision = metrics.precision_score(y_test, y_pred, average='weighted')
    model_recall = metrics.recall_score(y_test, y_pred, average='weighted')
    model_f1 = metrics.f1_score(y_test, y_pred, average='weighted')

    test_acc.append(model_accuracy)
    test_prec.append(model_precision)
    test_rec.append(model_recall)
    test_f1.append(model_f1)
    print(f">>> Fold-{fold} accuracy of MLP model = {(100*model_accuracy):.4f} %\n")

print('RESULTS: K-Fold Cross Validation')
print(f'Average accuracy after {3}-fold cross validation = {np.mean(test_acc):.4f}')
print(f'Average precision after {3}-fold cross validation = {np.mean(test_prec):.4f}')
print(f'Average recall after {3}-fold cross validation = {np.mean(test_rec):.4f}')
print(f'Average f1-score after {3}-fold cross validation = {np.mean(test_f1):.4f}')

"""Classification Report"""

print(classification_report(y_test, y_pred, target_names=tag_set))
# print(classification_report(y_test, y_pred))

"""Confusion Matrix"""

plt.figure(figsize=(12,8))
cf = confusion_matrix(y_test, y_pred)
cf_plot = sns.heatmap(cf, annot=True, cmap='GnBu', xticklabels=tag_set, yticklabels=tag_set)
plt.title("CONFUSION MATRIX")
plt.xlabel('PREDICTED Labels')
plt.ylabel('TRUE Labels')
plt.show()

"""### Insights

- In the MLP classifier, the GloVe embedding model outperforms the Word2vec embedding model significantly.
- We can observe a class imbalance problem in the data. Some tags such as 'X' and 'PRT' have relatively lesser instances compared to other tags and thus are misclassified mostly.
- Punctuation marks tag **('.')** is misclassified in the Word2vec model since there is no pre-existing embedding associated with punctuation marks. Thus, the embedding will be generated randomly as per our implementation. On the contrary, this is not the case with GloVe model due to which there are only a few instances of misclassification for punctuation marks tag.
"""

# wv_model = KeyedVectors.load('embeddings/word2vec-google-news-300.kv')
# print('.' in wv_model.vocab)

# gl_model = KeyedVectors.load('embeddings/glove-wiki-gigaword-300.kv')
# print('.' in gl_model.vocab)

