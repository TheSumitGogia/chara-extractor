import argparse
import os
from collection import feature_extract

'''Feature extraction for character candidates.
'''

def handle_characters_command(args):
    print 'NOTE: file/directory names must be relative to current directory'
    print 'NOTE: features must be count, tag, cooc, or coref'
    absdir = os.path.abspath('.')
    featlist = args.featurelist
    cnlpdir = absdir + '/' + args.cnlpdir
    tokensdir = absdir + '/' + args.tokensdir
    canddir = absdir + '/' + args.canddir
    outdir = absdir + '/' + args.outdir
    readable = args.readable
    feature_extract.extract_char_features(featlist, cnlpdir, tokensdir, canddir, outdir, readable)

def handle_relations_command(args):
    print 'NOTE: file/directory names must be relative to current directory'
    print 'NOTE: use candidate files must have "character" in filename (defaults to SN labels folder)!'
    print 'NOTE: features must be count, tag, cooc, or coref'
    absdir = os.path.abspath('.')
    featlist = args.featurelist
    cnlpdir = abspath + '/' + args.cnlpdir
    tokensdir = abspath + '/' + args.tokensdir
    canddir = abspath + '/' + args.canddir
    outdir = abspath + '/' + args.outdir
    readable = args.readable
    feature_extract.extract_pair_features(featlist, cnlpdir, tokensdir, canddir, outdir, readable)

if __name__ == '__main__':

    # general access command
    parser = argparse.ArgumentParser(description='extract features for candidates')
    subparsers = parser.add_subparsers(help='sub-command information', dest='command')

    # characters subcommand argument parsing
    char_parser = subparsers.add_parser('character', help='extract single character candidate features')
    char_parser.add_argument('-f', '--featurelist', nargs='+', default=['count,tag,cooc,coref'], help='list of feature types to extract')
    char_parser.add_argument('-d', '--cnlpdir', default='data/corenlp', help='directory with CoreNLP files for raw texts')
    char_parser.add_argument('-t', '--tokensdir', default='data/tokens', help='directory with CoreNLP tokens for raw texts')
    char_parser.add_argument('-c', '--canddir', default='data/candidates', help='directory with candidate files for raw texts')
    char_parser.add_argument('-o', '--outdir', default='data', help='directory to place feature files')
    char_parser.add_argument('r', '--readable', action='store_true', help='output readable file features also')

    # relations subcommand argument parsing
    char_parser = subparsers.add_parser('relation', help='extract relation candidate pair features')
    char_parser.add_argument('-f', '--featurelist', nargs='+', default=['count,tag,cooc,coref'], help='list of feature types to extract')
    char_parser.add_argument('-d', '--cnlpdir', default='data/corenlp', help='directory with CoreNLP files for raw texts')
    char_parser.add_argument('-t', '--tokensdir', default='data/tokens', help='directory with CoreNLP tokens for raw texts')
    char_parser.add_argument('-c', '--canddir', default='data/labels', help='directory with candidate files for raw texts')
    char_parser.add_argument('-o', '--outdir', default='data', help='directory to place feature files')
    char_parser.add_argument('r', '--readable', action='store_true', help='output readable file features also')

    args = parser.parse_args()
    if args.command == 'character':
        handle_characters_command(args)
    elif args.command == 'relation':
        handle_relations_command(args)
