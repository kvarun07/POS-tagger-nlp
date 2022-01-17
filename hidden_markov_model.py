# Import required libraries

import string
import collections
import itertools
import numpy as np 
import nltk
import sklearn
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import random

np.random.seed(5)

"""## Preprocessing"""

# Pre process dataset

lines = []

with open('Brown_tagged_train.txt', 'r') as file:
  lines = file.readlines()

lines = np.random.shuffle(lines)

for i in range(0, len(lines)):
  lines[i] = lines[i].strip().split()

def custom_split(token):
  ind = token.rfind('/')
  return (token[ind+1:], token[:ind])

tagged_sents = []

def load_dataset(lines):
  tagged_sents.clear()

  for line in lines:
    tagged_sents.append(('S', 'S'))
    tagged_sents.extend(list(map(custom_split, line)))
    tagged_sents.append(('E', 'E'))

  return tagged_sents

"""## Calculate Probabilities"""

cpd_tag_word, cpd_tag = None, None
unique_tags = None

def calc_prob(tag_word_pair):
  global cpd_tag_word, cpd_tag, unique_tags

  # generate conditional frequency
  cfd_tag_word = nltk.ConditionalFreqDist(tag_word_pair)

  # find conditional probability distribution
  cpd_tag_word = nltk.ConditionalProbDist(cfd_tag_word, nltk.MLEProbDist)

  # print("The probability of a verb (VB) being 'try' is", cpd_tag_word["VB"].prob("try"))
  tag_seq = list(map(lambda item : item[0], tag_word_pair))
  unique_tags = set(tag_seq)

  # Get all bigrams
  bigrams = nltk.bigrams(tag_seq)

  cfd_tag = nltk.ConditionalFreqDist(bigrams)
  cpd_tag = nltk.ConditionalProbDist(cfd_tag, nltk.MLEProbDist)

  # print(tag_seq)
  # print(unique_tags)

"""##Viterbi Algorithm"""

def predict_tags(sample_sent):
  """ sample_sent is a list of tuples (word, tag) """
  sample_sent_tokens = list(map(lambda x: x[1], sample_sent))


  # Initialize variables for viterbi

  # contains dictionaries
  # at index i, dictionary for i-th word in sample sentence
  # dictionary: (key, value) -> (tag, prob)
  # prob is the probability for the best pos tagging till position i, ending with tag
  best = []

  # same structure as best, stores thre previous best tag
  prev = []

  # Initialize best and prev
  best.append({})
  prev.append({})

  for tag in unique_tags:
    best[0][tag] = cpd_tag["S"].prob(tag) * cpd_tag_word[tag].prob(sample_sent_tokens[0])
    prev[0][tag] = "S"

  # variable to store the current best answer
  curr_best = max(best[0].keys(), key=lambda tag: best[0][tag])

  for token in sample_sent_tokens:
    new_dict_for_best = {}
    new_dict_for_prev = {}

    # find best pos tagging
    for tag in unique_tags:
      best_prevtag_prob = ('.', 0)

      for p_tag in best[-1].keys():
        prob = best[-1][p_tag] * cpd_tag[p_tag].prob(tag) * cpd_tag_word[tag].prob(token)
        if(prob > best_prevtag_prob[1]):
          best_prevtag_prob = (p_tag, prob)
      
      new_dict_for_best[tag] = best_prevtag_prob[1]
      new_dict_for_prev[tag] = best_prevtag_prob[0]
    
    best.append(new_dict_for_best)
    prev.append(new_dict_for_prev)

  prev_viterbi = best[-1]
  prev_best = max(prev_viterbi.keys(),
                      key=lambda prev_tag: prev_viterbi[prev_tag] * cpd_tag[prev_tag].prob("E"))
  
  # backtrack and find the best pos tagging
  prev_tag = prev_best
  ans = []

  for index in range(0, len(sample_sent_tokens)):
    ans.append(((prev_tag), sample_sent_tokens[-(index+1)]))
    prev_tag = prev[-(index+1)][prev_tag]

  ans.reverse()
  return ans

"""## Evaluate"""

"""K-Fold Cross Validation (K=3)"""

kf = KFold(n_splits=3)
print(kf)
kf.get_n_splits(lines)

mlp_model = None
history = None
y_pred = None
y_test = None
fold = 0

test_acc = []
test_prec = []
test_rec = []
test_f1 = []

lines = np.asarray(lines)
for train_index, test_index in kf.split(lines):
  fold += 1
  print(f'\nFOLD - {fold}')
  X_train, X_test = lines[train_index], lines[test_index]

  tagged_sents = load_dataset(X_train)
  tagged_sents_test = load_dataset(X_test)

  y_train = [i[0] for i in tagged_sents]
  y_test = [i[0] for i in tagged_sents_test if (i[0] != 'S' and i[0] != 'E')]

  # Call model
  calc_prob(tagged_sents)

  # Predictions
  y_pred = []
  for line in X_test:
    l = predict_tags(list(map(custom_split, line)))
    pred = [i[0] for i in l]
    y_pred.extend(pred)

  # Accuracy
  model_accuracy = metrics.accuracy_score(y_test, y_pred)
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

print(classification_report(y_test, y_pred))

"""Confusion Matrix"""

plt.figure(figsize=(12,8))
cf = confusion_matrix(y_test, y_pred)
cf_plot = sns.heatmap(cf, annot=True, cmap='GnBu')
plt.title("CONFUSION MATRIX")
plt.xlabel('PREDICTED Labels')
plt.ylabel('TRUE Labels')
plt.show()

"""## Statistics of tagset"""

# Number of words in the dataset
no_of_words = 0

# Minimum lenght of a sentence in the dataset
min_length = len(lines[0])

# Maximum length of a sentence in the dataset
max_length = len(lines[0])

# Sum of lengths of sentences in the dataset
total_length = 0

for line in lines:
  no_of_words += len(line)
  min_length = min(min_length, len(line))
  line_no += 1
  max_length = max(max_length, len(line))
  total_length += len(line)

# Average length of sentences in the dataset
avg_length = total_length / len(lines)

# Number of unique tags in the dataset
no_of_tags = len(unique_tags)

print('Number of Words:', no_of_words)
print('Minimum length of a sentence:', min_length)
print('Maximum length of a sentence:', max_length)
print('Average length of a sentence:', total_length / len(lines))
print('Number of tags in the dataset:', no_of_tags)

dataset = load_dataset(lines)
dict_tag_word = {}
dict_word_tag = {}

for tag, word in dataset:
  if tag == 'S' or tag == 'E':
    continue
  
  if tag in dict_tag_word.keys():
    dict_tag_word[tag] += 1
  else:
    dict_tag_word[tag] = 1
  
  if word in dict_word_tag.keys():
    dict_word_tag[word].add(tag)
  else:
    dict_word_tag[word] = {tag,}

print('Number of elements for each tag:', dict_tag_word)

ambiguous = {}
for word in dict_word_tag.keys():
  if len(dict_word_tag[word]) > 1:
    ambiguous[word] = dict_word_tag[word]

print('Ambiguous words:', ambiguous)
print('Number of ambiguous words:', len(ambiguous))
