from subprocess import check_call, check_output
import os, re, unicodedata, string, subprocess, operator
from chara.resolve.disambiguation import *
from optparse import OptionParser
import networkx as nx
from networkx.algorithms import bipartite

def process_features(book, temp):
    print book
    try:
        if temp:
            with open('features/%s_char_features.temp' % book, 'w') as f:
                command = ['python', 'feature_parser.py', '-f', 'raw_nlp/%s.txt.xml' % book, '-rf', 'tokens/%s.txt' % book, '-d', '-o', 'features']
                check_call(command, stderr=subprocess.STDOUT, stdout=f)
            return True
        else:
            command = ['python', 'feature_parser.py', '-f', 'raw_nlp/%s.txt.xml' % book, '-rf', 'tokens/%s.txt' % book, '-o', 'features']
            check_call(command, stderr=subprocess.STDOUT)
            return True
    except subprocess.CalledProcessError:
        print "ERROR command %s" % " ".join(command)
        return False

def strict_fuzzy_match(s1, s2):
    s1 = s1[0].lower() + s1[1:]
    s2 = s2[0].lower() + s2[1:]
    if s1 == 'grey' and s2 == 'gray':
        return True
    if s1 == 'gray' and s2 == 'grey':
        return True
    # ignore plural macthing
    if s2 == s1 + 's' or s1 == s2 + 's':
        return 0
    if (s1 not in ALL_TITLES and s2 not in ALL_TITLES):
        if s2.endswith('.') and s1.startswith(s2[:-1]):
            return 0.5
        if fuzz.ratio(s1, s2) >= 80:
            return fuzz.ratio(s1, s2)/100.0
    return 0

def strict_fuzzy_contains_tuple(t_outer, t_inner):
    if len(t_inner) == 0:
        return 0
    if len(t_inner) == 1 and t_inner[0] in ALL_TITLES:
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
            else:
                # penealize for the tokens that is not in t_inner
                score_sum -= 0.1
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
            if ALL_TITLES[cand[0]] != ALL_TITLES[ocand[0]]:
                return 0
            else:
                return strict_fuzzy_contains_tuple(ocand[1:], cand[1:])

        score = strict_fuzzy_contains_tuple(ocand, cand[1:])
        if score > 0:
            first_name = ocand[0].lower()
            if cand[0] in OTHER_TITLES:
                return score + 0.2
            elif first_name in gender_dict:
                if gender_dict[first_name] == 'MALE' and cand[0] in MALE_TITLES:
                    return score + 0.2
                elif gender_dict[first_name] == 'FEMALE' and cand[0] in FEMALE_TITLES:
                    return score + 0.2
            else:
                return score
    return 0

def match_to_any_names(character_names, cand):
    return max([strict_fuzzy_match_reference(character_name, cand) for character_name in character_names])

def match_candidates_and_characters(characters, candidates):
    matches = dict([(character, {}) for character in characters])

    # generate a graph that connects candidates and characters that match
    G = nx.Graph()
    G.add_nodes_from(characters, bipartite=0)
    G.add_nodes_from(candidates, bipartite=1)
    for character in characters:
        names = []
        for name in [character] + characters[character]:
            names.append(tuple(name.replace(',', ' ').replace('\'s ', ' \'s ').replace('s\'', 's \' ').split()))
        for cand in candidates:
            score = match_to_any_names(names, cand)
            if score > 0:
                G.add_edge(character, cand, weight=score)
        # if don't find any match, try the other direction
        # sparknote character name might be contained by some candidate names
        if len(matches[character]) == 0 and len(candidates) > 0:
            scores = [strict_fuzzy_match_reference(cand, names[0]) for cand in candidates]
            index, score = max(enumerate(scores), key=operator.itemgetter(1))
            if score > 0:
                G.add_edge(character, candidates[index], weight=score)

    max_matching = nx.max_weight_matching(G, maxcardinality=True)

    return (max_matching, G)

def get_char_features_from_file(book, temp, feature_directory):
    # get features from file
    file = '%s/%s_char_features.%s' % (feature_directory, book, 'temp' if temp else 'txt')
    try:
        with open(file) as f:
            lines = f.readlines()
    except:
        print '%s does not exist!' % file
        return None
    lines = [feature.strip() for feature in lines if feature.strip() != ""]
    if temp:
        features = eval("{" + ", ".join(lines[1:]) + "}")
    else:
        features = eval(lines[0])
    return features.keys()

def get_sparknote_characters_from_file(book, sndir='data/sparknotes/extracted'):
    try:
        with open(sndir + '/%s_characters.txt' % book) as f:
            line = f.readline()
    except:
        print sndir + '/%s_characters.txt does not exist!' % book
        return None
    characters = eval(line)
    return characters

def get_sparknote_relations_from_file(book, sndir='sparknotes/extracted'):
    try:
        with open(sndir + '/%s_relations.txt' % book) as f:
            relationships = eval(f.readline())
            return relationships
    except:
        print sndir + '/%s_relations.txt does not exist!' % book
        return None

def label_book(book, temp, feature_directory, unique, from_sparknote=True):
    print 'Labeling %s' % book

    # reading features and annotations
    candidates = get_char_features_from_file(book, temp, feature_directory)
    characters = get_sparknote_characters_from_file(book)
    relations= get_sparknote_relations_from_file(book)

    if candidates == None or characters == None or relations == None:
        return

    # matching candidates to sparknote characters
    (max_matching, G) = match_candidates_and_characters(characters, candidates)

    # labeling candidates
    if unique:
        cand_labels = dict([(cand, "") for cand in candidates])
        for cand in candidates:
            if cand in max_matching:
                cand_labels[cand] = max_matching[cand]
    else:
        cand_labels = dict([(cand, 0) for cand in candidates])
        for cand in candidates:
            if G.degree(cand) > 0:
                cand_labels[cand] = 1

    if from_sparknote:
        pair_candidates = [cand for cand in candidates if cand_labels[cand]==1]
    else:
        # TODO get candidates from output of character extractor
        pass

    # labeling pair
    # only between candidates that do not map to the same character
    pair_labels = dict([((cand1, cand2), 0) \
            for cand1 in pair_candidates for cand2 in pair_candidates \
            if not (cand1 == cand2)])
    for cand1 in candidates:
        for cand2 in candidates:
            if cand1 != cand2:
                char_set1 = G.neighbors(cand1)
                char_set2 = G.neighbors(cand2)
                if any([True for char1 in char_set1 for char2 in char_set2 if \
                        (char1, char2) in relations]):
                    pair_labels[(cand1, cand2)] = 1

    name = "%s_non_unique_characters.txt" % book if not unique else "%s_characters.txt" % book
    with open('labels/%s' % (name), 'w') as f:
        f.write(str(cand_labels))
    name = "%s_non_unique_relations.txt" % book if not unique else "%s_relations.txt" % book
    with open(write_dir + '/%s' % (name), 'w') as f:
        f.write(str(pair_labels))

    unresolved = []
    for character in characters:
        if character in max_matching:
            if verbose:
                print "%s: %s among %s" % (character, max_matching[character], str(G.neighbors(character)))
        else:
            unresolved.append(character)
            if G.degree(character) > 0:
                print "Unresolve %s with matched candidates %s" % (character, G.neighbors(character))
    print "Unresolved %s" % (unresolved)

    perc = len(unresolved)*1.0/len(characters)
    print "Unresolved percentage %f" % perc
    return perc

def label_all(
        indir='extracted',
        canddir='candidates',
        outdir='labels',
        unique=False):
    all_books = os.listdir(canddir)
    all_books = [book[:-4] for book in all_books]
    perc = []
    write_dir = outdir
    for book in all_books:
        p = label_book(book, False, canddir, unique)
        if p is not None:
            perc.append(p)
    if len(perc) > 0:
        print "Unresolved percentage range [%f, %f] mean %f" %(min(perc), max(perc), sum(perc)/len(perc))


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-b", "--book", dest="book", help="which book to process", default="all")
    parser.add_option("-f", "--process_features", dest="features", action="store_true", default=False)
    parser.add_option("-t", "--temp_features", dest="temp", action="store_true", default=False)
    parser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False)
    parser.add_option("-d", "--feature_directory", dest="feature_directory", default="features")
    parser.add_option("-u", "--unique", dest="unique", action="store_true", default=False)

    (options, args) = parser.parse_args()
    verbose = options.verbose
    if options.book == 'all':
        all_books = os.listdir('raw_texts')
        all_books = [book[:-4] for book in all_books]
    else:
        all_books = [options.book]

    write_dir = 'labels'
    perc = []
    for book in all_books:
        to_label = True
        if options.features:
            to_label = process_features(book, options.temp)
        if to_label:
            p = label_book(book, options.temp, options.feature_directory, options.unique)
            if p is not None:
                perc.append(p)
    if len(perc) > 0:
        print "Unresolved percentage range [%f, %f] mean %f" %(min(perc), max(perc), sum(perc)/len(perc))
