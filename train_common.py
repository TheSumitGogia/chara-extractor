from optparse import OptionParser
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
import numpy as np
import os, random, re, time
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier 
from sklearn.feature_extraction import DictVectorizer
from labeling import get_sparknote_characters_from_file

FEATURES_DIR = 'features/'
LABELS_DIR = 'labels/'

def generate_train_test(pct_train, seed):
    if seed == None:
        seed = int(time.time()) 
    print "seed " + seed
    random.seed(seed)
    books = set(map(lambda f: f.split('_')[0], \
                    filter(lambda f: not f.endswith('.swp'),
                            os.listdir(FEATURES_DIR))))
    train_set = random.sample(books, int(len(books) * pct_train))
    test_set = books.difference(train_set)
    return train_set, test_set

# Read single character features from file
def read_features(book, dir, extension, filter):
    file = dir + book + extension
    try:
        with open(file) as f:
            features_dict = eval(f.readline())
    except:
        print "%s does not exist" % file
        return 
    if len(filter) == 0:
        return features_dict
    filtered_features = filter(lambda s: re.match(filter, s), \
                            features_dict.itervalues().next())
    return {cand: {feature: features_dict[cand][feature] \
                    for feature in filtered_features}
            for cand in features_dict}
    

# DictVectorizer for a single book
def vectorize(features_dict):
    v = DictVectorizer(sparse=False)
    return v.fit_transform(features_dict.values())   

# Sparknotes labels
def get_labels(book, features_dict, dir, extension):
    with open(dir + book + extension) as f:
        d = eval(f.readline())
        labels_dict = {k: int(d[k] != '' and d[k] != 0) for k in d}
    return np.array(map(labels_dict.__getitem__, features_dict.keys()))

def get_data(books, features_dir, features_ext, labels_dir, labels_ext, feature_filter, print_features=False):
    Xs = []
    ys = []
    cands = []
    for book in books:
        features_dict = read_features(book, features_dir, features_ext, feature_filter)
        if print_features:
            print features_dict.itervalues().next().keys()
            print_features=False
        if features_dict is not None:
            X = vectorize(features_dict)
            y = get_labels(book, features_dict, labels_dir, labels_ext)
            Xs.append(X)
            ys.append(y)
            cands.extend(features_dict.keys())
    return np.vstack(Xs), np.hstack(ys), np.array(cands)

def precision(y_pred, y_true):
    if sum(y_pred) == 0:
        return 1
    return float(sum(y_pred & y_true)) / sum(y_pred)

def recall(y_pred, y_true):
    return float(sum(y_pred & y_true)) / sum(y_true)

def evaluate_books(clf, books, scaler, evaluate):
    perfs = []
    for book in books:
        perf = evaluate(clf, book, scaler)
        if perf is not None:
            perfs.append(perf)
    perfs = np.array(perfs)
    mean = np.mean(perfs, axis=0)
    return mean
