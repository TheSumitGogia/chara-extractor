import argparse
import os
from labeling import *
'''
import data_collect
import process_sparknote
import labeler
'''

'''Sparknotes Label extraction and application to character candidates.'''

def handle_collect_command(args):
    print 'NOTE: file/directory names must be relative to main project directory'
    absdir = os.path.abspath('.')
    outdir = args.outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    data_collect.collect(outdir + '/descriptions')

def handle_process_command(args):
    print 'NOTE: file/directory names must be relative to main project directory'
    print 'NOTE: with test option, dir must point to annotation tags, not raw texts'
    print 'NOTE: outdir ignored if test option selected'
    sndir = args.sndir
    rawdir = args.rawdir
    outdir = args.outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    test = args.test
    if (test):
        process_sparknote.test(sndir, rawdir)
    else:
        process_sparknote.process_all(sndir, rawdir, outdir)

def handle_label_command(args):
    print 'NOTE: file/directory names must be relative to main project directory'
    indir = args.indir
    canddir = args.canddir
    outdir = args.outdir
    unique = args.unique

    labeler.label_all(indir, canddir, outdir, unique)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='extract and apply labels from Sparknotes')
    subparsers = parser.add_subparsers(help='sub-command information', dest='command')

    # collect subcommand argument parsing
    collect_parser = subparsers.add_parser('collect', help='collect books and chara descriptions')
    collect_parser.add_argument('-o', '--outdir', default='data/sparknotes', help='output directory')

    # collect subcommand argument parsing
    process_parser = subparsers.add_parser('process', help='run chara and relation extraction')
    process_parser.add_argument('-s', '--sndir', default='data/sparknotes/descriptions', help='sparknotes descriptions directory')
    process_parser.add_argument('-d', '--dir', default='data/raw', help='directory with book files')
    process_parser.add_argument('-o', '--outdir', default='data/sparknotes/extracted', help='output directory')
    process_parser.add_argument('-t', '--test', action='store_true', help='run manual annotation comparison test')

    # label subcommand argument parsing
    label_parser = subparsers.add_parser('label', help='run automated tagger for candidates using Sparknotes')
    label_parser.add_argument('-i', '--indir', default='data/sparknotes/extracted', help='Sparknotes data directory')
    label_parser.add_argument('-c', '--canddir', default='data/candidates', help='candidates directory')
    label_parser.add_argument('-o', '--outdir', default='data/labels', help='directory to output label files')
    label_parser.add_argument('-u', '--unique', action='store_true', help='whether to label unique/disamb candidates')
    label_parser.add_argument('-t', '--test', action='store_true', help='print labeling results for verification')

    args = parser.parse_args()
    if args.command == 'collect':
        handle_collect_command(args)
    elif args.command == 'process':
        handle_process_command(args)
    elif args.command == 'label':
        handle_label_command(args)
