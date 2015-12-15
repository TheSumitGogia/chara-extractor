from optparse import OptionParser
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
import numpy as np
import os, random, re, time
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.feature_extraction import DictVectorizer
import chara.labeling
from chara.labeling.labeler import get_sparknote_characters_from_file

def generate_train_test(pct_train, seed, dir):
    if seed == None:
        seed = int(time.time())
    #print "seed ", seed
    random.seed(seed)
    books = set(map(lambda f: f.split('_')[0], \
                    filter(lambda f: not f.endswith('.swp'),
                            os.listdir(dir))))
    train_set = random.sample(books, int(len(books) * pct_train))
    test_set = books.difference(train_set)
    return train_set, test_set

def read_features(book, dir, extension, feature_filter):
    #print book
    file = dir + '/' + book + extension

    with open(file) as f:
        lines = [line.strip() for line in f.readlines() if line.strip() != '']
    data_split = [1 if line[0] == '(' else 0 for line in lines]
    num_data = sum(data_split)
    num_features = data_split[1:].index(1)
    # some pair features does not contain coref
    if not any([True if line.startswith('coref:') else False for line in lines[1:num_features+1]]):
        num_features += 1
    feature_values = np.zeros([num_data, num_features], dtype=np.float64)
    features = []
    names = []
    line_num = 0
    for line in lines:
        line_num += 1
        if line[0] == '(':
            idx = len(names)
            feature_idx = 0
            name = eval(line)
            names.append(name)
        else:
            tokens = line.split(':')
            if tokens[0] != 'coref':
                if idx == 0:
                    features.append(tokens[0])
                else:
                    assert features[feature_idx] == tokens[0], "%s vs %s feature %d line %d" % (features[feature_idx], tokens[0], feature_idx, line_num)
                feature_values[idx][feature_idx] = float(tokens[1])
                feature_idx+=1
            else:
                feature_values[idx][num_features-1] = float(tokens[1])

    if len(feature_filter) == 0:
        return (feature_values, features, names)
    filtered_features = filter(lambda s: re.match(feature_filter, s), features)
    valid_features = [i for i in range(len(features)) if features[i] in filtered_features]
    feature_values = feature_values[:,valid_features]
    assert feature_values.shape == (len(names), len(filtered_features))
    return (feature_values, filtered_features, names)

# Sparknotes labels
def get_labels(book, dir, extension):
    with open(dir + '/' + book + extension) as f:
        d = eval(f.readline())
        labels_dict = {k: int(d[k] != '' and d[k] != 0) for k in d}
    return labels_dict

def get_data(books, features_dir, features_ext, labels_dir, labels_ext, feature_filter, print_features=False):
    Xs = []
    ys = []
    cands = []
    for book in books:
        (feature_values, features, cands) = read_features(book, features_dir, features_ext, feature_filter)
        if print_features:
            print features
            print_features=False
        labels_dict = get_labels(book, labels_dir, labels_ext)
        # filter the unlabeled data
        labeled_cands = [cand for cand in cands if cand in labels_dict]
        rows = [i for i in range(len(cands)) if cands[i] in labeled_cands]
        X = feature_values[rows,:]
        y = np.array(map(labels_dict.__getitem__, labeled_cands))
        assert X.shape[0] == y.shape[0]
        Xs.append(X)
        ys.append(y)
        cands.extend(labeled_cands)
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

