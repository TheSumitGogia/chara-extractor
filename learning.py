import argparse
from training import *
import scipy.io as sio

''' Character and relation classifier training.
'''

def handle_translate_command(args):
    pass

def handle_datasplit_command(args):
    pass

def handle_train_command(args):
    pass

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
    '''
    # datasplit command argument parsing
    split_parser = subparsers.add_parser('datasplit', help='split features, labels into train/test sets')
    split_parser.add_argument('-d', '--datadir', default='data', help='directory to get data files from')
    split_parser.add_argument('-o', '--outdir', default='data', help='directory to output split data files to')
    '''

    # train command argument parsing
    train_parser = subparsers.add_parser('train', help='train binary classifier on features and labels')
    train_parser.add_argument('-t', '--type', default='char', choices=['char', 'rel'], help='whether to train character or relation classifier')
    train_parser.add_argument('-m', '--model', default='svm')
    train_parser.add_argument('-b', '--bias', default=2)
    train_parser.add_argument('--degree', default=2)
    train_parser.add_argument('-r', '--ratio', default='0.7', help='train test split ratio')
    train_parser.add_argument('-k', '--kernel', default='rbf')
    train_parser.add_argument('-f', '--featdir', default='features', help='directory to get features from')
    train_parser.add_argument('-l', '--labeldir', default='labels', help='directory to get labels from')
    train_parser.add_argument('-x', '--filter', default=[], help='feature filter, regex string list')
    train_parser.add_argument('-o', '--outdir', default='classifiers', help='output directory for classifier parameters')

    args = parser.parse_args()
    '''
    if args.command == 'translate':
        handle_translate_command(args)
    elif args.command == 'datasplit':
        handle_datasplit_command(args)
    '''
    if args.command == 'train':
        handle_train_command(args)
