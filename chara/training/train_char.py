from chara.resolve.disambiguation import find_unique_characters
from evaluation import evaluate_candidates
from chara.labeling.labeler import get_sparknote_characters_from_file
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.externals import joblib
from matplotlib import pyplot as plt
from optparse import OptionParser
import numpy as np
import os, random, re
from train_common import *

FEATURES_DIR = 'features'
LABELS_DIR = 'labels'
CHAR_FEATURES_EXTENSION = '_char_features_readable.txt'
CHAR_LABELS_EXTENSION = '_non_unique_characters.txt'
CHAR_FEATURE_FILTER = ''
n_estimators = None

DEFAULT_FILTER = [
  #'count'
  #'cooc.*', 'book.*', 'count.*', '.*cap'
  #'coref_shorter_count_norm_char'
]

def evaluate_char(clf, book, scaler = None, baseline = False):
    print book
    X, y, cands, features = get_data([book], FEATURES_DIR, CHAR_FEATURES_EXTENSION, LABELS_DIR, CHAR_LABELS_EXTENSION, CHAR_FEATURE_FILTER)
    if scaler != None:
        X = scaler.transform(X)
    if baseline:
        counts = X[:,features.index('count')]
        lengths = X[:,features.index('length')]
        one_grams_counts = counts
        one_grams_counts[lengths != 1] = 0
        top_one_grams = (-one_grams_counts).argsort()[:20]
        bi_grams_counts = counts
        bi_grams_counts[lengths != 2] = 0
        top_bi_grams = (-bi_grams_counts).argsort()[:10]
        tri_grams_counts = counts
        bi_grams_counts[lengths != 3] = 0
        top_tri_grams = (-tri_grams_counts).argsort()[:5]

        select = list(top_one_grams) + list(top_bi_grams) + list(top_tri_grams)
        y_pred = np.array([1 if i in select else 0 for i in range(len(counts))])
    else:
        y_pred = clf.predict(X)
    precision_rate = precision(y_pred, y)
    recall_rate = recall(y_pred, y)
    print 'Non unqiue Precision:', precision(y_pred, y), 'Non unique Recall:', recall(y_pred, y)

    characters = get_sparknote_characters_from_file(book)
    # TODO: option for suppressing this output
    if characters is not None:
        cands_pred = cands[y_pred==1]
        cands_pred_unique = find_unique_characters(cands_pred)
        unresolved, duplicate, invalid= evaluate_candidates(characters, cands_pred_unique)

        unresolve_rate = len(unresolved)*1.0/len(characters)
        duplicate_rate = len(duplicate)*1.0/len(cands_pred_unique) if len(cands_pred_unique) != 0 else 0
        invalid_rate = len(invalid)*1.0/len(cands_pred_unique) if len(cands_pred_unique) != 0 else 0
        if verbose:
            print "True"
            print characters
            print "Pred"
            print cands_pred_unique
            print "Unresolved"
            print unresolved
            print "Duplicate"
            print duplicate
            print "Invalid"
            print invalid
        print "Unresolve: %f, duplicate %f, invalid: %f" % (unresolve_rate, duplicate_rate, invalid_rate)

        return [unresolve_rate, duplicate_rate, invalid_rate, precision_rate, recall_rate]
    return None

def get_char_data(books, print_features=False):
    return get_data(books, FEATURES_DIR, CHAR_FEATURES_EXTENSION, LABELS_DIR, CHAR_LABELS_EXTENSION, CHAR_FEATURE_FILTER, print_features)

# `train` is a function that takes in training data and output clf
def train_and_test(train_books, test_books, train, scale=True):
    X_train, y_train, cands_train, features = get_char_data(train_books, True)
    X_test, y_test, cands_test, features = get_char_data(test_books)

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
    train_perf = evaluate_books(clf, train_books, scaler, evaluate_char)

   # print performance for testing books
    print "\n"
    print "--------------Testing data-------------"
    test_perf = evaluate_books(clf, test_books, scaler, evaluate_char)

    print 'Tran Non-unique Precision:', precision(y_train_pred, y_train), 'Non-unique Recall:', recall(y_train_pred, y_train)
    print 'Test Non-unique Precision:', precision(y_test_pred, y_test), 'Recall:', recall(y_test_pred, y_test)
    print "Train Unresolve: %f, duplicate %f, invalid: %f" % (train_perf[0], train_perf[1], train_perf[2])
    print "Test Overall Unresolve: %f, duplicate %f, invalid: %f" % (test_perf[0], test_perf[1], test_perf[2])
    return clf, scaler

def get_and_save_data(books, outdir='clfdata'):
    X, y, cands, features = get_char_data(books, True)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    np.save(outdir + '/char_features.npy', X)
    np.save(outdir + '/char_labels.npy', y)
    bookfile = open(outdir + '/' + 'books.txt', 'w')
    bookfile.write('\n'.join(books))
    bookfile.close()

# `train` is a function that takes in training data and output clf
def train_and_save(train_books, train, clf_name='clfparams', scale=True):
    X_train, cands_train, features = get_char_data(train_books, True)

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

    clf = train(X_train, y_train)
    if not os.path.exists(clf_fname):
        os.makedirs(clf_fname)
    joblib.dump(clf, clf_fname + '/' + 'classifier.pkl')
    joblib.dump(scaler, clf_fname + '/' + 'scaler.pkl')
    trainfile = open(clf_fname + '/' + 'train_books.txt', 'w')
    trainfile.write('\n'.join(train_books))
    trainfile.close()

def train_from_file_and_save(train, train_dir='clfdata', clf_fname='char_clf', scale=True):
    train_bfile = open(train_dir + '/books.txt', 'r')
    train_books = train_bfile.readlines()
    train_books = [book.strip() for book in train_books]
    X_train = np.load(train_dir + '/char_features.npy')
    y_train = np.load(train_dir + '/char_labels.npy')
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
    clf = train(X_train, y_train)
    if not os.path.exists(clf_fname):
        os.makedirs(clf_fname)
    joblib.dump(clf, clf_fname + '/' + 'classifier.pkl')
    joblib.dump(scaler, clf_fname + '/' + 'scaler.pkl')
    trainfile = open(clf_fname + '/' + 'train_books.txt', 'w')
    trainfile.write('\n'.join(train_books))

def evaluate_clf_from_file(clf_dirname, testbooks=None):
    books = set(map(lambda f: f.split('_')[0], \
                    filter(lambda f: not f.endswith('.swp'),
                            os.listdir(FEATURES_DIR))))
    train_bkfile = open(clf_dirname + '/train_books.txt', 'r')
    train_books = train_bkfile.readlines()
    train_books = set([book.strip() for book in train_books])
    if not testbooks:
        test_books = books.difference(train_books)
    else:
        test_books = set(testbooks)

    clf = joblib.load(clf_dirname + '/classifier.pkl')
    scaler = joblib.load(clf_dirname + '/scaler.pkl')

    if not testbooks:
        train_perf = evaluate_books(clf, train_books, scaler, evaluate_char)
        print 'Train Non-unique Precision:', train_perf[3], 'Non-unique Recall:', train_perf[4]
        print "Train Unresolve: %f, duplicate %f, invalid: %f" % (train_perf[0], train_perf[1], train_perf[2])
    test_perf = evaluate_books(clf, test_books, scaler, evaluate_char)
    print 'Test Non-unique Precision:', test_perf[3], 'Recall:', test_perf[4]
    print "Test Overall Unresolve: %f, duplicate %f, invalid: %f" % (test_perf[0], test_perf[1], test_perf[2])

def evaluate_baseline(test_books=None):
    books = set(map(lambda f: f.split('_')[0], \
                    filter(lambda f: not f.endswith('.swp'),
                            os.listdir(FEATURES_DIR))))
    if not test_books:
        test_books = books
    else:
        test_books = set(test_books)

    test_perf = evaluate_books(None, test_books, None, evaluate_char, True)
    print 'Test Non-unique Precision:', test_perf[3], 'Recall:', test_perf[4]
    print "Test Overall Unresolve: %f, duplicate %f, invalid: %f" % (test_perf[0], test_perf[1], test_perf[2])

# traning methods for different training models
def train_svm(X, y):
    clf = svm.SVC(kernel=kernel, degree=degree, class_weight=class_weight)
    clf.fit(X, y)
    return clf

# random forest
def train_rf(X, y):
    clf = None
    if not n_estimators:
        clf = RandomForestClassifier(class_weight=class_weight)
    else:
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth = max_depth, class_weight=class_weight)
    clf.fit(X,y)
    return clf

# extra random forest
def train_erf(X, y):
    clf = ExtraTreesClassifier(n_estimators=n_estimators, max_depth = max_depth, class_weight=class_weight)
    clf.fit(X,y)
    return clf

# ada_boost
def train_ada_boost(X, y):
    clf = AdaBoostClassifier(n_estimators=n_estimators)
    clf.fit(X,y)
    return clf

# grad boosted trees
def train_grad_boost(X, y):
    clf = GradientBoostingClassifier(n_estimators=n_estimators)
    clf.fit(X,y)
    return clf

def set_filters(filters):
    global CHAR_FEATURE_FILTER
    CHAR_FEATURE_FILTER = '|'.join('^%s$' % f for f in eval(filters))

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
    parser.add_option("--seed", dest="seed", default='None')

    ################## hyper-parameters for models #################
    # degree for polynomial kernel for svm
    parser.add_option("--degree", dest="degree", default=2)
    # a bias term to adjust tradeoff between precision and recall,
    # the higher bias, the higher recall and lower precision
    parser.add_option("-b", "--bias", dest="bias", default=1)
    # for random forest, adaboost, and gradboost
    # suggest 10 for random forest, 100 for adaboost and gradboost
    parser.add_option("-n", "--n_estimators", dest="n_estimators", default=10)
    # for random forest
    parser.add_option("--max_depth", dest="max_depth", default='None')

    # parse options
    (options, args) = parser.parse_args()
    FEATURES_DIR=options.feature_directory
    FEATURES_EXTENSION=options.feature_extension
    LABELS_DIR=options.label_directory
    filters = eval(options.feature_filter)
    verbose = options.verbose

    CHAR_FEATURE_FILTER = '|'.join('^%s$' % f for f in filters)

    # set traning options
    kernel = options.kernel
    degree = int(options.degree)
    train_books, test_books = generate_train_test(float(options.train_ratio), options.seed, FEATURES_DIR)
    train_method = locals()['train_%s' % options.model]
    class_weight = {1:float(options.bias), 0:1}
    n_estimators = int(options.n_estimators)
    max_depth = eval(options.max_depth)

    (clf, scaler) = train_and_test(train_books, test_books, train_method)
