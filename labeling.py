from subprocess import check_call, check_output
import os, re, unicodedata, string, subprocess
from disambiguation import *
from optparse import OptionParser
import networkx as nx
from networkx.algorithms import bipartite

def process_features(book):
    print book
    try:
        with open('features/%s_char_features.temp' % book, 'w') as f:
            command = ['python', 'feature_parser.py', '-f', 'raw_nlp/%s.txt.xml' % book, '-rf', 'tokens/%s.txt' % book, '-d', '-o', 'features']
            check_call(command, stderr=subprocess.STDOUT, stdout=f)
        return True
    except subprocess.CalledProcessError:
        print "ERROR command %s" % " ".join(command)
        return False

def strict_fuzzy_match(s1, s2):
    s1 = s1[0].lower() + s1[1:]
    s2 = s2[0].lower() + s2[1:]
    # ignore plural macthing
    if s2 == s1 + 's' or s1 == s2 + 's':
        return 0
    if (s1 not in ALL_TITLES and s2 not in ALL_TITLES) and fuzz.ratio(s1, s2) >= 80:
        return fuzz.ratio(s1, s2)/100.0
    else:
        return 0

def strict_fuzzy_contains_tuple(t_outer, t_inner):
    if len(t_inner) == 0:
        return 0
    inner_idx=0
    score_sum = 0.0
    # firstname matches firstname or lastname matches lastname
    if strict_fuzzy_match(t_outer[0], t_inner[0]) > 0 or strict_fuzzy_match(t_outer[-1], t_inner[-1]) > 0:
        for outer_idx in range(len(t_outer)):
            score = strict_fuzzy_match(t_outer[outer_idx], t_inner[inner_idx])
            if score > 0:
                # perfer firstname matching
                score_sum += score + (0.5 if outer_idx==0 else 0)
                inner_idx+=1
            if inner_idx == len(t_inner):
                return score_sum
    return 0

def strict_fuzzy_match_reference(ocand, cand):
    if len(ocand) < len(cand):
        return 0

    if ocand == cand:
        return len(cand) + 1
    
    # first try contains_tuple
    score = strict_fuzzy_contains_tuple(ocand, cand)
    if score > 0:
        return score
    # then try title
    if cand[0] in ALL_TITLES: 
        if ocand[0] in ALL_TITLES:
            if cand[0] != ocand[0]:
                return 0
            else:
                return strict_fuzzy_contains_tuple(ocand[1:], cand[1:])
        
        score = strict_fuzzy_contains_tuple(ocand, cand[1:])
        if score > 0:
            first_name = ocand[0].lower()
            if first_name in gender_dict:
                if gender_dict[first_name] == 'MALE' and cand[0] in MALE_TITLES:
                    return score + 0.2
                elif gender_dict[first_name] == 'FEMALE' and cand[0] in FEMALE_TITLES:
                    return score + 0.2
            else:
                return 0
                #print (ocand, cand)
                #print '%s not in firstname dict' % first_name
    return 0

def find_matched_character_names(character_names, cand):
    match = {}
    for character_name in character_names:
        score = strict_fuzzy_match_reference(character_name, cand)
        if score > 0:
            match[character_name] = score
    # find no match in character that contains cand
    # try the other direction, find characters that 
    # is contained by cand
    '''
    if len(character) == 0:
        for character in characters:
            score = strict_fuzzy_match_reference(cand, character)
            if score > 0:
                match[character] = score
    '''
    return match

def match_candidates_and_characters(characters, all_candidates):
    labels = dict([(cand, "") for cand in all_candidates])
    references = {}
    # references is a map that maps difference names we got from sparknotes 
    # to characters
    for character in characters:
        for name in characters[character] + [character]:
            tokens = tuple(name.replace(',', ' ').replace('\'s ', ' \'s ').replace('s\'', 's \' ').split())
            references[tokens] = character
    names = references.keys()

    # matches is a map that maps characters to a list of candidates that can 
    # represent this character
    matches = dict([(character, {}) for character in characters])
    for cand in all_candidates:
        possible_matches = find_matched_character_names(names, cand)
        for match in possible_matches:
            character = references[match]
            if cand in matches[character]:
                matches[character][cand] = max(possible_matches[match], matches[character][cand])
            else:
                matches[character][cand] = possible_matches[match]
    
    # generate a graph from matches and run max matching
    G = nx.Graph()
    G.add_nodes_from(characters, bipartite=0)
    G.add_nodes_from(all_candidates, bipartite=1)
    for character in matches:
        for cand in matches[character]:
            G.add_edge(character, cand, weight=matches[character][cand])
   
    max_matching = nx.max_weight_matching(G, maxcardinality=True)
    unresolved = []
    for character in characters:
        if character in max_matching:
            labels[character] = max_matching[character]
            if verbose:
                print "%s: %s among %s" % (character, labels[character], str(matches[character]))
        else:
            if len(matches[character]) > 0:
                print "Unresolve %s with matched candidates %s" % (character, str(matches[character]))
            unresolved.append(character)
    print "Unresolved %s" % (unresolved)
    
    with open('labels/%s_characters.txt' % book, 'w') as f:
        f.write(str(labels))

    perc = len(unresolved)*1.0/len(characters)
    print "Unresolved percentage %f" % perc
    return perc
'''
    for character in characters:
        if len(matches[character]) == 0:
            unresolved.append(character)
        else:
            max_score = 0
            best_cand = ""
            candidates_score = matches[character]
            for cand in candidates_score:
                score = candidates_score[cand]
                if score > max_score:
                    best_cand = cand
                    max_score = score
            labels[best_cand] = character
            if verbose:
                print "%s: %s among %s" % (character, best_cand, str(matches[character]))
    if len(unresolved) > 0:
        print "Unresolved %s" % unresolved 
        '''

def label_book(book, temp):
    print 'Labeling %s' % book
    
    # get features from file
    file = 'features/%s_char_features.%s' % (book, 'temp' if temp else 'txt')
    try:
        with open(file) as f:
            features = f.readlines()
    except:
        print '%s does not exist!' % file
        return -1
    features = [feature.strip() for feature in features if feature.strip() != ""]
    if temp:
        features = eval("{" + ", ".join(features[1:]) + "}")
    else:
        features = eval(features[0])

    all_candidates = features.keys()
    try:
        with open('sparknotes/%s_characters.txt' % book) as f:
            characters = eval(f.readline())
    except:
        print 'sparknotes/%s_characters.txt does not exist!' % book
        return -1
    return match_candidates_and_characters(characters, all_candidates)

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-b", "--book", dest="book", help="which book to process", default="all")
    parser.add_option("-f", "--process_features", dest="features", action="store_true", default=False)
    parser.add_option("-t", "--temp_features", dest="temp", action="store_true", default=False)
    parser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False)
    (options, args) = parser.parse_args()
    verbose = options.verbose
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
            p = label_book(book, options.temp)
            if p >= 0:
                perc.append(p)
    if len(perc) != 0:
        print "Unresolved percentage range [%f, %f] mean %f" %(min(perc), max(perc), sum(perc)/len(perc))
