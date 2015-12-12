import argparse
from training import *

'''Evaluate classifiers for character and relation extraction
'''

def handle_clf_command(args):
    pass

def handle_test_command(args):
    pass

if __name__ == '__main__':

    # general access command
    parser = argparse.ArgumentParser(description='evaluate classifiers for characters and binary relations')
    subparsers = parser.add_subparsers(help='sub-command information')

    # clf subcommand argument parsing
    clf_parser = subparsers.add_parser('clf', help='get precision/recall for classifier')
    clf_parser.add_argument('-d', '--datadir', default='data', help='directory with test features, labels')
    clf_parser.add_argument('-n', '--nonuniq', action='store_true', help='get straight classifier prec/recall')
    clf_parser.add_argument('-d', '--disamb', action='store_true', help='get disambiguated classifier prec/recall')

    # test subcommand argument parsing
    test_parser = subparsers.add_parsers('test', help='output found characters for input books')
    test_parser.add_argument('-b', '--books', default=False, help='book file/folder to test on, off by default since slow')
    test_parser.add_argument('-c', '--candidates', default='candidates', help='file/folder with candidates')
    test_parser.add_argument('-t', '--tokens', default='rawtokens', help='file/folder with raw text tokens (CoreNLP)')
    test_parser.add_argument('-n', '--corenlp', default='rawcorenlp', help='file/folder with raw text CoreNLP files')
