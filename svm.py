from disambiguation import find_unique_characters
from evaluation import evaluate_candidates
from labeling import get_sparknote_characters_from_file
from sklearn import preprocessing
from sklearn import svm
from sklearn.feature_extraction import DictVectorizer
from matplotlib import pyplot as plt
import numpy as np
import os
import random
import re

FEATURES_DIR = 'running_env/features/'
FEATURES_DIR = 'features_good/features/'
CHAR_FEATURES_EXTENSION = '_char_features.txt'
LABELS_DIR = 'labels/'
LABELS_EXTENSION = '_non_unique_characters.txt'

filters = [
  #'cooc.*', 'book.*', 'count.*', '.*cap'
  #'coref_shorter_count_norm_char'
]
FEATURE_FILTER = '|'.join('^%s$' % f for f in filters)

def generate_train_test(pct_train):
  books = set(map(lambda f: f.split('_')[0], \
                  filter(lambda f: not f.endswith('.swp'),
                         os.listdir(FEATURES_DIR))))
  train_set = random.sample(books, int(len(books) * pct_train))
  test_set = books.difference(train_set)
  return train_set, test_set

# Read single character features from file
def read_char_features(book):
  with open(FEATURES_DIR + book + CHAR_FEATURES_EXTENSION) as f:
    features_dict = eval(f.readline())
    if len(filters) == 0:
      return features_dict
    filtered_features = filter(lambda s: re.match(FEATURE_FILTER, s), \
                               features_dict.itervalues().next())
    return {cand: {feature: features_dict[cand][feature] \
                     for feature in filtered_features}
              for cand in features_dict}

# DictVectorizer for a single book
def vectorize(features_dict):
  v = DictVectorizer(sparse=False)
  print features_dict.itervalues().next().keys()
  return v.fit_transform(features_dict.values())   

# Sparknotes labels
def get_labels(book, features_dict):
  with open(LABELS_DIR + book + LABELS_EXTENSION) as f:
    d = eval(f.readline())
    labels_dict = {k: int(d[k] != '' and d[k] != 0) for k in d}
  return np.array(map(labels_dict.__getitem__, features_dict.keys()))

def get_data(books):
  Xs = []
  ys = []
  cands = []
  for book in books:
    features_dict = read_char_features(book)
    X = vectorize(features_dict)
    y = get_labels(book, features_dict)
    Xs.append(X)
    ys.append(y)
    cands.extend(features_dict.keys())
  return np.vstack(Xs), np.hstack(ys), np.array(cands)

def precision(y_pred, y_true):
  return float(sum(y_pred & y_true)) / sum(y_pred)

def recall(y_pred, y_true):
  return float(sum(y_pred & y_true)) / sum(y_true)

if __name__ == '__main__':
  train, test = generate_train_test(0.7)
  X_train, y_train, cands_train = get_data(train)
  X_test, y_test, cands_test = get_data(test)

  scaler = preprocessing.StandardScaler()
  X_train_fit = scaler.fit_transform(X_train)
  X_test_fit = scaler.transform(X_test)

  clf = svm.SVC(probability=True)
  clf.fit(X_train_fit, y_train)
  y_train_pred = clf.predict(X_train_fit)
  y_test_pred = clf.predict(X_test_fit)
  print 'Precision:', precision(y_train_pred, y_train), 'Recall:', recall(y_train_pred, y_train)
  print 'Precision:', precision(y_test_pred, y_test), 'Recall:', recall(y_test_pred, y_test)

  cands_true = get_sparknote_characters_from_file(
  cands_train_pred = cands_train[y_train_pred == 1]
  cands_train_pred_unique = find_unique_characters(cands_train_pred)
  unresolved_train, duplicate_train, invalid_train = evaluate_candidates(cands_train_true_unique, cands_train_pred_unique)
  print unresolved_train, duplicate_train, invalid_train
  print len(unresolved_train), len(duplicate_train), len(invalid_train)

  #plt.plot(X_train_fit, y_train, 'o')
  #plt.ylim(-0.2, 1.2)
  #plt.show()
