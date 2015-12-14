import argparse
from chara.training import *
import scipy.io as sio

''' Character and relation classifier training.
'''

def handle_getdata_command(args):
    tp = args.type
    split = args.split
    seed = args.seed
    featdir = args.featdir
    labeldir = args.labeldir
    filt = args.filter
    outdir = args.outdir
    if tp == 'char':
        train_char.FEATURES_DIR = featdir + '/characters'
        train_char.LABELS_DIR = labeldir
        train_books, test_books = train_common.generate_train_test(split, seed, featdir + '/characters')
        train_char.set_filters(filt)
        train_char.get_and_save_data(train_books, outdir)

    elif tp == 'rel':
        train_pair.FEATURES_DIR = featdir + '/relations'
        train_pair.LABELS_DIR = labeldir
        train_books, test_books = train_common.generate_train_test(split, seed, featdir + '/relations')
        train_pair.set_filters(filt)
        train_pair.get_and_save_data(train_books, outdir)

def handle_train_command(args):
    tp = args.type
    model = args.model
    bias = args.bias
    degree = args.degree
    kernel = args.kernel
    outdir = args.outdir
    datadir = args.datadir

    if tp == 'char':
        train_method = vars(train_char)['train_%s' % model]
        train_char.class_weight = {1:float(bias), 0:1}
        train_char.kernel = kernel
        train_char.degree = degree
        train_char.train_from_file_and_save(train_method, datadir, outdir)
    elif tp == 'rel':
        train_method = vars(train_pair)['train_%s' % model]
        train_pair.class_weight = {1:float(bias), 0:1}
        train_pair.kernel = kernel
        train_pair.degree = degree
        train_pair.train_from_file_and_save(train_method, datadir, outdir)

if __name__ == '__main__':

    # general access command
    parser = argparse.ArgumentParser(description='setup and train character and relation classifiers')
    subparsers = parser.add_subparsers(help='sub-command information', dest='command')

    # not in place yet because of some code inflexibility...
    '''
    # translate subcommand argument parsing
    trans_parser = subparsers.add_parser('translate', help='translate dictionary of features into feature matrix')
    trans_parser.add_argument('-t', '--type', default='char', choices=['char', 'rel'], help='whether features are for characters or relations')
    trans_parser.add_argument('-f', '--featuredir', default='features', help='directory containing all features')
    trans_parser.add_argument('-o', '--outdir', default='data', help='directory to put data files in')
    '''

    # not in place yet because of some code inflexibility
    # datasplit command argument parsing
    getdata_parser = subparsers.add_parser('getdata', help='get formatted train/test data for sklearn classifiers')
    getdata_parser.add_argument('-t', '--type', default='char', choices=['char', 'rel'], help='whether data is for characters or relations')
    getdata_parser.add_argument('-f', '--featdir', default='data/features', help='directory with character and relation features')
    getdata_parser.add_argument('-l', '--labeldir', default='data/labels', help='directory with character and relation labels')
    getdata_parser.add_argument('-s', '--split', default=0.7, type=float, help='ratio of training data for data split')
    getdata_parser.add_argument('-x', '--filter', default='[]', help='feature filter, regex string list')
    getdata_parser.add_argument('--seed', default=None, help='seed for random split')
    getdata_parser.add_argument('-o', '--outdir', default='data/training/clfdata', help='directory to output split data files to')

    # train command argument parsing
    train_parser = subparsers.add_parser('train', help='train binary classifier on features and labels')
    train_parser.add_argument('-t', '--type', default='char', choices=['char', 'rel'], help='whether to train character or relation classifier')
    train_parser.add_argument('-m', '--model', default='svm', help='classification model')
    train_parser.add_argument('-b', '--bias', default=2, type=int, help='positive class weight')
    train_parser.add_argument('--degree', default=2, help='degree of polynomial kernel if used')
    train_parser.add_argument('-d', '--datadir', default='data/training/clfdata', help='directory with training data')
    train_parser.add_argument('-k', '--kernel', default='rbf', help='kernel for kernel-based models')
    train_parser.add_argument('-o', '--outdir', default='data/classifiers/clf', help='output directory for classifier parameters')

    args = parser.parse_args()
    '''
    if args.command == 'translate':
        handle_translate_command(args)
    '''
    if args.command == 'getdata':
        handle_getdata_command(args)
    if args.command == 'train':
        handle_train_command(args)
