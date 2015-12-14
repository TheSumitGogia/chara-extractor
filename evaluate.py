import argparse
from chara.training import *

'''Evaluate classifiers for character and relation extraction
'''

def handle_quant_command(args):
    tp = args.type
    clfdir = args.clfdir
    featdir = args.featdir
    labeldir = args.labeldir

    if tp == 'char':
        train_char.FEATURES_DIR = featdir + '/characters'
        train_char.LABELS_DIR = labeldir
        train_char.verbose = False
        train_char.evaluate_clf_from_file(clfdir)
    if tp == 'rel':
        train_pair.FEATURES_DIR = featdir + '/relations'
        train_pair.LABELS_DIR = labeldir
        train_pair.verbose = False
        train_pair.evaluate_clf_from_file(clfdir)

def handle_qual_command(args):
    tp = args.type
    clfdir = args.clfdir
    featdir = args.featdir
    labeldir = args.labeldir
    books = args.books

    if tp == 'char':
        train_char.FEATURES_DIR = featdir + '/characters'
        train_char.LABELS_DIR = labeldir
        train_char.verbose = True
        train_char.evaluate_clf_from_file(clfdir, books)
    if tp == 'rel':
        train_pair.FEATURES_DIR = featdir + '/relations'
        train_pair.LABELS_DIR = labeldir
        train_pair.verbose = True
        train_pair.evaluate_clf_from_file(clfdir, books)

if __name__ == '__main__':

    # general access command
    parser = argparse.ArgumentParser(description='evaluate classifiers for characters and binary relations')
    subparsers = parser.add_subparsers(help='sub-command information', dest='command')

    # clf subcommand argument parsing
    clf_parser = subparsers.add_parser('quant', help='get precision/recall for classifier')
    clf_parser.add_argument('-t', '--type', default='char', help='test characters or relations')
    clf_parser.add_argument('-c', '--clfdir', default='data/classifiers/clfparams', help='directory with classifier params')
    clf_parser.add_argument('-f', '--featdir', default='data/features', help='directory with features')
    clf_parser.add_argument('-l', '--labeldir', default='data/labels', help='directory with labels')

    # test subcommand argument parsing
    test_parser = subparsers.add_parser('qual', help='output found characters for input books')
    test_parser.add_argument('-t', '--type', default='char', help='test characters or relations')
    test_parser.add_argument('-b', '--books', nargs='+', help='book names or book list file')
    test_parser.add_argument('-c', '--clfdir', default='data/classifiers/clfparams', help='directory with classifier params')
    test_parser.add_argument('-f', '--featdir', default='data/features', help='file/folder with features')
    test_parser.add_argument('-l', '--labeldir', default='data/labels', help='directory with labels')

    args = parser.parse_args()
    if args.command == 'quant':
        handle_quant_command(args)
    elif args.command == 'qual':
        handle_qual_command(args)
