from subprocess import check_call, check_output
import os, re, unicodedata, string, subprocess
from disambiguation import *
from optparse import OptionParser

def process_features(book):
    print book
    try:
        command = ['python', 'feature_parser.py', '-f', 'raw_nlp/%s.txt.xml' % book, '-rf', 'raw_texts/%s.txt' % book, '-o', 'features', '-n', '[100,100,50]', '-cn', '50']
        check_output(command, stderr=subprocess.STDOUT)
        return True
    except subprocess.CalledProcessError:
        print "ERROR extracting feature %s" % book
        return False

def label_book(book):
    print 'Labeling %s' % book
    try:
        with open('features/%s_char_features.txt' % book) as f:
            features = eval(f.readline())
            all_candidates = features.keys()
        labels = dict([(cand, "") for cand in all_candidates])
    except:
        print 'features/%s_char_features.txt does not exist!' % book
        return False
    try:
        with open('sparknotes/%s_characters.txt' % book) as f:
            characters = eval(f.readline())
    except:
        print 'sparknotes/%s_characters.txt does not exist!' % book
        return False
    references = {}
    for character in characters:
        for name in characters[character] + [character]:
            tokens = tuple(name.replace(',', ' ').split())
            references[tokens] = character
    names = references.keys()
    matches = dict([(character, set()) for character in characters])
    for cand in all_candidates:
        possible_matches =  partial_reference(names, cand) + \
                            title_resolution(names, cand)
        if cand in references:
            possible_matches.append(cand)
        for match in possible_matches:
            matches[references[match]].add(cand)
    unresolved = []
    for character in characters:
        if len(matches[character]) == 0:
            unresolved.append(character)
        else:
            max_length = (0, 0)
            best_cand = ""
            for cand in matches[character]:
                length = (len(cand), sum([len(token) for token in cand]))
                if length > max_length:
                    best_cand = cand
                    max_length = length
            labels[best_cand] = character
            print "%s: %s among %s" % (character, best_cand, str(matches[character]))
    if len(unresolved) > 0:
        print "Unresolved %s" % unresolved
    
    with open('labels/%s_characters.txt' % book, 'w') as f:
        f.write(str(labels))

    perc = len(unresolved)*1.0/len(characters)
    print "Unresolved percentage %f" % perc
    return perc

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-b", "--book", dest="book", help="which book to process", default="all")
    parser.add_option("-f", "--features", dest="features", action="store_true", default=False)
    (options, args) = parser.parse_args()
    if options.book == 'all':
        all_books = os.listdir('raw_texts')
        with open("bad_books.txt", 'r') as f:
            bad_books = f.readlines()
        
        bad_books = set([book[:-1] for book in bad_books if book.endswith('.txt\n')])
        print bad_books

        all_books = [book[:-4] for book in all_books if book not in bad_books]
    else:
        all_books = [options.book]

    perc = []
    for book in all_books:
        to_label = True
        if options.features:
            to_label = process_features(book)
        if to_label:
            if label_book(book):
                perc.append(label_book(book))
    if len(perc) != 0:
        print "Unresolved percentage range [%f, %f] mean %f" %(min(perc), max(perc), sum(perc)/len(perc))
