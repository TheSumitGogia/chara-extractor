from disambiguation import find_unique_characters
from evaluation import evaluate_candidates
from labeling import get_sparknote_characters_from_file
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from matplotlib import pyplot as plt
from optparse import OptionParser
import numpy as np
import os, random

FEATURES_DIR = 'features/'
CHAR_FEATURES_EXTENSION = '_char_features.txt'
LABELS_DIR = 'labels/'
LABELS_EXTENSION = '_non_unique_characters.txt'

DEFAULT_FILTER = [
  #'cooc.*', 'book.*', 'count.*', '.*cap'
  #'coref_shorter_count_norm_char'
]

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
    return v.fit_transform(features_dict.values())   

# Sparknotes labels
def get_labels(book, features_dict):
    with open(LABELS_DIR + book + LABELS_EXTENSION) as f:
        d = eval(f.readline())
        labels_dict = {k: int(d[k] != '' and d[k] != 0) for k in d}
    return np.array(map(labels_dict.__getitem__, features_dict.keys()))

def get_data(books, print_features=False):
    Xs = []
    ys = []
    cands = []
    for book in books:
        features_dict = read_char_features(book)
        if print_features:
            print features_dict.itervalues().next().keys()
            print_features=False
        X = vectorize(features_dict)
        y = get_labels(book, features_dict)
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

def evaluate(clf, book, scaler = None):
    print book
    X, y, cands = get_data([book])
    if scaler != None:
        X = scaler.transform(X)

    y_pred = clf.predict(X)
    print 'Non unqiue Precision:', precision(y_pred, y), 'Non unique Recall:', recall(y_pred, y)

    characters = {}
    if get_sparknote_characters_from_file(book, characters):
        cands_pred = cands[y_pred==1]
        cands_pred_unique = find_unique_characters(cands_pred)
        unresolved, duplicate, invalid= evaluate_candidates(characters, cands_pred_unique)
        if verbose:
            print "Unresolved"
            print unresolved
            print "Dupliate"
            print duplicate
            print "Invalid"
            print invalid
        
        unresolve_rate = len(unresolved)*1.0/len(characters)
        duplicate_rate = len(duplicate)*1.0/len(cands_pred_unique) if len(cands_pred_unique) != 0 else 0
        invalid_rate = len(invalid)*1.0/len(cands_pred_unique) if len(cands_pred_unique) != 0 else 0
        print "Unresolve: %f, duplicate %f, invalid: %f" % (unresolve_rate, duplicate_rate, invalid_rate)

        return [unresolve_rate, duplicate_rate, invalid_rate]
    return 0

def evaluate_books(clf, books, scaler):
    perfs = []
    for book in books:
        perf = evaluate(clf, book, scaler)
        if perf > 0:
            perfs.append(perf)
    perfs = np.array(perfs)
    mean = np.mean(perfs, axis=0)
    return mean

# `train` is a function that takes in training data and output clf
def train_and_test(train_books, test_books, train, scale=True):
    X_train, y_train, cands_train = get_data(train_books, print_features=True)
    X_test, y_test, cands_test = get_data(test_books)
    
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    clf = train(X_train, y_train)
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    # print performance for training books
    print "--------------Traning data-------------"
    train_perf = evaluate_books(clf, train_books, scaler)
   
   # print performance for testing books 
    print "\n"
    print "--------------Testing data-------------"
    test_perf = evaluate_books(clf, test_books, scaler)
    
    print 'Tran Non-unique Precision:', precision(y_train_pred, y_train), 'Non-unique Recall:', recall(y_train_pred, y_train)
    print 'Test Non-unique Precision:', precision(y_test_pred, y_test), 'Recall:', recall(y_test_pred, y_test)
    print "Train Unresolve: %f, duplicate %f, invalid: %f" % (train_perf[0], train_perf[1], train_perf[2])
    print "Test Overall Unresolve: %f, duplicate %f, invalid: %f" % (test_perf[0], test_perf[1], test_perf[2])
    return clf, scaler

# traning methods for different training models
def train_svm(X, y):
    clf = svm.SVC(kernel=kernel, degree=degree, class_weight=class_weight)
    clf.fit(X, y)
    return clf

# random forest
def train_rf(X, y):
    clf = RandomForestClassifier(n_estimators=n_estimators, class_weight=class_weight)
    clf.fit(X,y)
    return clf

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-d', '--feature_dir', help="feature directory", dest="feature_directory", default=FEATURES_DIR)
    parser.add_option('-e', '--feature_extension', help="feature extension", dest="feature_extension", default=CHAR_FEATURES_EXTENSION)
    parser.add_option('-l', '--label_dir', help='label directory', dest='label_directory', default=LABELS_DIR)
    parser.add_option('-f', '--feature_filter', help='feature filter', dest='feature_filter', default=str(DEFAULT_FILTER))
    parser.add_option('-s', '--train_test_split', help='Ratio of training books', dest='train_ratio', default='0.7')
    parser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False)
    parser.add_option("-m", "--model", dest="model")
    parser.add_option("-k", "--kernel", dest="kernel", default='rbf')
    # degree for polynomial kernel for svm
    parser.add_option("--degree", dest="degree", default=2)
    # a bias term to adjust tradeoff between precision and recall,
    # the higher bias, the higher recall and lower precision
    parser.add_option("-b", "--bias", dest="bias", default=1)
    # for random forest
    parser.add_option("-n", "--n_estimators", dest="n_estimators", default=10)
    
    # parse options
    (options, args) = parser.parse_args()
    FEATURES_DIR=options.feature_directory
    CHAR_FEATURES_EXTENSION=options.feature_extension
    LABELS_DIR=options.label_directory
    filters = eval(options.feature_filter)
    verbose = options.verbose

    FEATURES_FILTER = '|'.join('^%s$' % f for f in filters)
    
    # set traning options
    kernel = options.kernel
    degree = int(options.degree)
    train_books, test_books = generate_train_test(float(options.train_ratio))
    train_method = locals()['train_%s' % options.model]
    class_weight = {1:float(options.bias), 0:1}
    n_estimators = int(options.n_estimators)

    (clf, scaler) = train_and_test(train_books, test_books, train_method)
