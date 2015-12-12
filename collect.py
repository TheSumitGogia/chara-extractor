import argparse
import os
from collection import dataparse

''' Character extraction un-annotated data collection from Project Gutenberg.
'''

'''
def handle_crossref_command(args):
    print 'NOTE: file/directory names are abs, or relative to current directory'
    print 'NOTE: blacklist and booklist must be file with each line containing a Sparknotes book identifier...'
    index = args.index
    booklist = args.booklist
    blacklist = args.blacklist
    outdir = args.outdir

    crossref.crossref(index, booklist, blacklist, outdir)

def handle_download_command(args):
    print 'NOTE: file/directory names are abs, or relative to current directory'
    print 'NOTE: good, uglyfix have books on lines, format: [snname|gutname|author|gutid]'
    absdir = os.path.abspath('.')
    goodfile = absdir + '/' + args.goodfile
    uglyfixfile = absdir + '/' + args.uglyfixfile if args.uglyfixfile else None
    outdir = absdir + '/' + args.outdir

    get_text.download(goodfile, uglyfixfile, outdir)
'''

def handle_parse_command(args):
    print 'NOTE: file/directory names are abs, or relative to current directory'
    print 'NOTE: Must have Stanford CoreNLP dir in environment variable CORE_NLP'
    absdir = os.path.abspath('.')
    raw = absdir + '/' + args.rawtext
    outdir = absdir + '/' + args.outdir

    nlpdir, tokensdir = dataparse.parse_corenlp(raw, outdir)
    dataparse.parse_candidates(nlpdir, tokensdir, outdir)

if __name__ == '__main__':

    # general access command
    parser = argparse.ArgumentParser(description='collect training samples from Project Gutenberg for character extraction')
    subparsers = parser.add_subparsers(help='sub-command information', dest='command')

    # crossref subcommand argument parsing
    cref_parser = subparsers.add_parser('crossref', help='search index for good, bad, and questionable files')
    cref_parser.add_argument('-i', '--index', default='data/gutenberg/index.txt', help='the Project Gutenberg index file')
    cref_parser.add_argument('-b', '--booklist', default='data/sparknotes/booklist.txt', help='the list of books to search for')
    cref_parser.add_argument('-bl', '--blacklist', default='data/gutenberg/blacklist.txt', help='the list of books to avoid')
    cref_parser.add_argument('-o', '--outdir', default='data/gutenberg', help='directory to output file lists into')

    # download subcommand argument parsing
    dl_parser = subparsers.add_parser('download', help='get raw texts from Project Gutenberg')
    dl_parser.add_argument('-g', '--goodfile', default='data/gutenberg/good.txt', help='file containing books definitely in Gutenberg')
    dl_parser.add_argument('-u', '--uglyfixfile', default=None, help='file containing questionable books edited to match in Gutenberg')
    dl_parser.add_argument('-o', '--outdir', default='data/raw', help='folder to place raw book texts in')

    # parse subcommand argument parsing
    parse_parser = subparsers.add_parser('parse', help='run Stanford CoreNLP and candidate extraction on raw texts')
    parse_parser.add_argument('-f', '--rawtext', default='data/raw', help='Project Gutenberg raw text directory or file')
    parse_parser.add_argument('-o', '--outdir', default='data/', help='Output directory for corenlp, tokens, candidates')

    args = parser.parse_args()
    if args.command == 'crossref':
        handle_crossref_command(args)
    elif args.command == 'download':
        handle_download_command(args)
    elif args.command == 'parse':
        handle_parse_command(args)
