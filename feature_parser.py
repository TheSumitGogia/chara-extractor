import xml.etree.ElementTree as ET
import operator
import argparse
import itertools as it
import disambiguation as dbg
import numpy as np
import sys, os, traceback
from nltk.corpus import wordnet as wn
import wordnet_hyponyms as wh
from collections import deque
from django.utils.encoding import smart_str
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize

SECTION_MAP = {'sentence': 'st', 'paragraph': 'pg', 'chapter': 'cp'}
def abbrv(section_name):
    return SECTION_MAP[section_name]

def load_token_lines(tokenf):
    token_lines = open(token_filename, 'r')
    token_lines = token_lines.readlines()
    token_lines = [token[:-1] for token in token_lines]
    token_lines = token_lines[2:]
    return token_lines

def section_process_token(token):
    token_t = smart_str(token[0].text.strip())
    nl_count = 0
    idx_start = tl_idx
    long_token_string = ''
    idx_change = 0
    while (smart_str(token_lines[tl_idx]) != token_t) and not token_t in long_token_string:
        test_token = token_lines[tl_idx]
        if test_token == '*NL*':
            nl_count += 1
        else:
            idx_change += 1
            long_token_string += test_token
        tl_idx += 1
    #if idx_change == 6:
    #    tl_idx = idx_start
    if token_lines[tl_idx] != token_t and token_t in long_token_string:
        #print 'hehe', token_t, long_token_string, tl_idx
        tl_idx -= 1


def section(tree, token_filename):
    print 'Starting sectioning...'
    markers = {}
    all_sentences = tree.getroot()[0][0]
    token_lines = open(token_filename, 'r')
    token_lines = token_lines.readlines()
    token_lines = [token[:-1] for token in token_lines]
    token_lines = token_lines[2:]
    tl_idx, sentence_idx, token_idx = 0, 0, 0
    st_markers, pg_markers, cp_markers = [], [], []
    for sentence in all_sentences:
        for tokens in sentence:
            st_markers.append(int(tokens[0][2].text))
            token_idx = 0
            for token in tokens:
                token_t = smart_str(token[0].text.strip())
                nl_count = 0
                try:
                    idx_start = tl_idx
                    long_token_string = ''
                    idx_change = 0
                    while (smart_str(token_lines[tl_idx]) != token_t) and not token_t in long_token_string:
                        test_token = token_lines[tl_idx]
                        if test_token == '*NL*':
                            nl_count += 1
                        else:
                            idx_change += 1
                            long_token_string += test_token
                        tl_idx += 1
                    #if idx_change == 6:
                    #    tl_idx = idx_start
                    if token_lines[tl_idx] != token_t and token_t in long_token_string:
                        #print 'hehe', token_t, long_token_string, tl_idx
                        tl_idx -= 1
                except:
                    print "FAIL"
                    print token_t, token[2].text
                    print tl_idx
                    print token_filename
                    return
                #print token_t, token_lines[tl_idx], tl_idx + 3, token[2].text
                tl_idx += 1
                if nl_count > 1 or len(pg_markers) == 0:
                    if nl_count > 2 or len(cp_markers) == 0:
                        cp_markers.append(int(token[2].text))
                    pg_markers.append(int(token[2].text))
                token_idx += 1
        sentence_idx += 1

    while (tl_idx < len(token_lines) and token_lines[tl_idx] == '*NL*'):
        tl_idx += 1
    # final check!!
    if (sentence_idx == len(all_sentences) and token_idx == len(all_sentences[sentence_idx-1][0])):
        if tl_idx == len(token_lines):
            #print "SUCCESS"
            hoho = 1
        else:
            print "FAIL"
            print len(token_lines), tl_idx
            print token_filename
    markers['sentence'] = st_markers
    markers['paragraph'] = pg_markers
    markers['chapter'] = cp_markers
    markers['book'] = st_markers[-1]
    print 'Finished sectioning!'

    return markers

def is_the(word):
    if word == 'the' or word == 'The' or word == 'THE':
        return True
    return False

# TODO: this method can be shortened easily, and should be
def get_candidates(tree, markers, cutoffs=[20, 20, 5], cp_cutoff=10):
    """Extract candidate character names for a book.

    Character names are defined as simple noun phrases consisting of at most
    one level of nesting (i.e. (NP (NP Det N) (Prep) (NP Det N))), and some
    capitalization. Candidate extraction consists of filtering the names
    fitting this description by frequency in different portions of the novel.

    Args:
        tree (ElementTree): The parsed Stanford NLP XML tree.
            This tree should be made with annotations tokenize, ssplit, pos,
            lemma, and ner.
        markers (dict[string -> list[int]]): Lists of character offsets.
            The dictionary maps section types to lists of character offsets
            from the beginning of the raw book text file. The character offsets
            indicating beginnings of new sections.

    Returns:
        dict[tuple[string] -> dict]: the mapping of candidates to features.

        The candidates are represented as tuples of token strings consecutive
        in the book. Features are represented by a dictionary from string
        feature names to their values.

        The features returned here are only 'count' and 'length', representing
        the frequency of the candidate in the text and length of the candidate
        respectively.
    """

    print 'Getting candidates...'
    # store candidates with counts per chapter
    ngrams = []
    # chapter offsets and tracking
    cp_markers = markers['chapter']
    total_length = int(markers['book'])
    cp_index = -1

    # stuff to indicate nesting of invalidity of candidate noun phrases
    bad_title_set = set(['Mr.', 'Mr', 'Mrs.', 'Ms.', 'Mrs', 'Ms', 'Miss'])
    bad_token_set = set(['Chapter', 'CHAPTER', 'PART', 'said', ',', "''", 'and', ';', '-RSB-', '-LSB-', '_', '--', '``', '.'])
    bad_np_tags = set(['CC', 'IN', 'TO', 'WDT', 'WP', 'WP$', 'WRB', 'UH', 'VB', 'VBD', 'VBP', 'VBZ', 'MD'])

    # TODO: load person set
    person_set = wh.enumerate_hyponyms(wn.synsets('person')[0])

    # loop through Stanford NLP tree, extracting initial candidate set
    root = tree.getroot()
    sentences = root[0][0]
    for sentence in sentences:
        for tokens in sentence:
            for token_idx in range(len(tokens)):
                # get current token information
                token = tokens[token_idx]
                word = token[0].text
                noun = (token[4].text.startswith('NN'))

                # handle new chapter
                if cp_index < len(cp_markers) - 1 and int(token[2].text) == cp_markers[cp_index + 1]:
                    ngrams.append({})

                # filter for candidates with last word noun and some capital
                if noun and (any(l.isupper() for l in word) or word in person_set):
                    # loop through previous words, adding to noun phrase
                    curr_idx = token_idx
                    word_list = [word]
                    first_tag = token[4].text
                    np_condition = True
                    exhausted = False
                    while np_condition and not exhausted:
                        # check if valid candidate and count
                        word_tuple = tuple(word_list)
                        if (word_tuple[0] in bad_token_set) or (len(word_tuple) == 1 and word in bad_title_set):
                            exhausted = True
                            break
                        if (first_tag.startswith('NN') or is_the(word_tuple[0])) and any(l.isupper() for l in word):
                            if curr_idx >= 1 and not any(l.isupper() for l in tokens[curr_idx - 1][0].text):
                                if word_tuple in ngrams[-1]:
                                    ngrams[-1][word_tuple] += 1
                                else:
                                    ngrams[-1][word_tuple] = 1
                        elif word in person_set and ((first_tag.startswith('NN') and len(word_list) > 1) or is_the(word_tuple[0])):
                            if word_tuple in ngrams[-1]:
                                ngrams[-1][word_tuple] += 1
                            else:
                                ngrams[-1][word_tuple] = 1

                        # continue adding previous words or moving up one layer
                        if curr_idx >= 1:
                            prev_token = tokens[curr_idx - 1]
                            prev_word = prev_token[0].text
                            prev_tag = prev_token[4].text
                            if prev_tag not in bad_np_tags:
                                word_list.insert(0, prev_word)
                                first_tag = prev_tag
                                curr_idx -= 1
                            else:
                                if curr_idx >= 2:
                                    pp_token = tokens[curr_idx - 2]
                                    pp_word = pp_token[0].text
                                    pp_tag = pp_token[4].text
                                    if pp_tag in bad_np_tags:
                                        exhausted = True
                                        break
                                    word_list.insert(0, prev_word)
                                    word_list.insert(0, pp_word)
                                    first_tag = pp_tag
                                    curr_idx -= 2
                                    np_condition = False
                                else:
                                    exhausted = True
                        else:
                            exhausted = True

                    # add second part of 2-level noun phrase
                    np_condition = True
                    while np_condition and not exhausted:
                        word_tuple = tuple(word_list)
                        if (word_tuple[0] in bad_token_set) or (len(word_tuple) == 1 and word in bad_title_set):
                            exhausted = True
                            break
                        if (first_tag.startswith('NN') or is_the(word_tuple[0])) and any(l.isupper() for l in word):
                            if word_tuple in ngrams[-1]:
                                ngrams[-1][word_tuple] += 1
                            else:
                                ngrams[-1][word_tuple] = 1
                        if curr_idx >= 1:
                            prev_token = tokens[curr_idx - 1]
                            prev_word = prev_token[0].text
                            prev_tag = prev_token[4].text
                            if prev_tag not in bad_np_tags:
                                word_list.insert(0, prev_word)
                                first_tag = prev_tag
                                curr_idx -= 1
                            else:
                                np_condition = False
                        else:
                            exhausted = True


    # add counts across all chapters
    norm_ngrams = {}
    for i in range(len(ngrams)):
        cp_ngrams = ngrams[i]
        for key in cp_ngrams:
            if key in norm_ngrams:
                norm_ngrams[key] += cp_ngrams[key]
            else:
                norm_ngrams[key] = cp_ngrams[key]

    dedup_candidates(tree, norm_ngrams)
    ngrams = get_general_count_feature(tree, norm_ngrams, markers)

    # get frequent candidates across whole book
    filtered_grams = []
    if cutoffs == 'flex':
        filtered_grams.extend(sorted(norm_ngrams.items(), key=operator.itemgetter(1)))
        filtered_grams = [gram for gram in filtered_grams if gram[1] > 2]
    else:
        for gram_size in range(1, 8):
            grams = { gram: norm_ngrams[gram] for gram in norm_ngrams.keys() if len(gram) == gram_size }
            sorted_grams = sorted(grams.items(), key=operator.itemgetter(1))
            if gram_size <= len(cutoffs):
                if len(sorted_grams) == 0: continue
                cutoff_idx = int(-1.0 * cutoffs[gram_size-1] / 100 * len(sorted_grams))
                pass_grams = sorted_grams[cutoff_idx:]
                # don't include them if they don't occur often
                pass_grams = [gram for gram in pass_grams if gram[1] > 2]
                filtered_grams.extend(pass_grams)
                filtered_grams.extend([(gram, grams[gram]) for gram in grams if grams[gram] > 10])
            else:
                # don't include them if they don't occur often
                pass_grams = [sorted_grams[idx] for idx in range(len(sorted_grams)) if sorted_grams[idx][1] > 3]
                filtered_grams.extend(pass_grams)
                filtered_grams.extend([(gram, grams[gram]) for gram in grams if grams[gram] > 10])

    # get frequent candidates per chapter
    if cutoffs == 'flex':
        for cp_idx in range(len(ngrams)):
            cp_ngrams = ngrams[cp_idx]
            filtered_grams.extend([(gram, norm_ngrams[gram]) for gram in cp_ngrams if cp_ngrams[gram] > 4])
    else:
        for cp_idx in range(len(ngrams)):
            cp_ngrams = ngrams[cp_idx]
            for cp_gram_size in range(1, 8):
                cp_grams = { gram: cp_ngrams[gram] for gram in cp_ngrams.keys() if len(gram) == cp_gram_size }
                sorted_cp_grams = sorted(cp_grams.items(), key=operator.itemgetter(1))
                if len(sorted_cp_grams) == 0: continue
                cutoff_idx = int(-1.0 * cp_cutoff / 100 * len(sorted_cp_grams))
                top_cp_grams = sorted_cp_grams[cutoff_idx:]
                pass_cp_grams = [(gram[0], norm_ngrams[gram[0]]) for gram in top_cp_grams if gram[1] > 3]
                filtered_grams.extend(pass_cp_grams)
                filtered_grams.extend([(gram, cp_grams[gram]) for gram in cp_grams if cp_grams[gram] > 5])

    # changing list of candidate tuples with counts to feature map
    candidates = {}
    total = 0
    for i in range(len(filtered_grams)):
        key = filtered_grams[i][0]
        count = filtered_grams[i][1]
        total += count
        candidates[key] = {
            'count': count,
            'length': len(key),
            'book_num_chars': total_length,
            'book_num_st': len(markers['sentence']),
            'book_num_pg': len(markers['paragraph']),
            'book_num_cp': len(markers['chapter'])
        }
    for key in candidates:
        total += candidates[key]['count']
        candidates[key]['count_norm_length'] = candidates[key]['count'] * 1.0 / total_length
        candidates[key]['count_norm_char'] = candidates[key]['count'] * 1.0 / total

    print 'Got {0} candidates!'.format(len(candidates.keys()))
    return candidates

def get_candidate_pairs(ngrams):
    pairs = {}
    for cand in candidates:
        cand1_feats = candidates[cand]
        for cand2 in candidates:
            if cand == cand2:
                continue
            cand2_feats = candidates[cand2]
            pair = (cand, cand2)
            pairs[pair] = {}
            pair_features = pairs[pair]
            for feature in cand1_feats:
                pair_features["1_" + feature] = cand1_feats[feature]
                pair_features["2_" + feature] = cand2_feats[feature]

def dedup_candidates(tree, ngrams):
    max_gram = max(map(lambda x: len(x), ngrams.keys()))
    sentences = tree.getroot()[0][0]

    left_set, right_set = set([]), set([])
    # loop through Stanford NLP tree, checking tags when candidates appear
    for sentence in sentences:
        for tokens in sentence:
            pwords = deque([""] * max_gram)
            nwords = None
            if len(tokens) <= max_gram:
                nwords = deque([token[0].text for token in tokens])
            else:
                nwords = deque([tokens[tidx][0].text for tidx in range(max_gram)])
            for token_idx in range(len(tokens)):
                token = tokens[token_idx]
                word, offset = token[0].text, token[2].text
                word_list = []

                pwords.popleft()
                pwords.append(word)

                # get candidates from pwords and add to appropriate dicts
                lword_list = []
                lbiggest = None
                for i in range(1, len(pwords)+1):
                    lword_list.insert(0, pwords[-i])
                    lword_tuple = tuple(lword_list)
                    if lword_tuple in ngrams:
                        lbiggest = lword_tuple

                if lbiggest is not None:
                    left_set.add(lbiggest)

                rword_list = []
                rbiggest = None
                for i in range(len(nwords)):
                    rword_list.append(nwords[i])
                    rword_tuple = tuple(rword_list)
                    if rword_tuple in ngrams:
                        rbiggest = rword_tuple

                if rbiggest is not None:
                    right_set.add(rbiggest)

                nwords.popleft()
                if token_idx + max_gram < len(tokens):
                    nwords.append(tokens[token_idx + max_gram][0].text)

    # union all section keys
    final_candidates = left_set.intersection(right_set)
    rem_list = []
    for ngram in ngrams:
        if ngram not in final_candidates:
            rem_list.append(ngram)
    for ngram in rem_list:
        ngrams.pop(ngram, None)

def get_general_count_feature(tree, ngrams, markers):
    cp_markers = markers['chapter']
    cp_counts = []
    cp_dict = None
    cp_idx = -1
    max_gram = max(map(lambda x: len(x), ngrams.keys()))
    root = tree.getroot()
    sentences = root[0][0]
    for sentence in sentences:
        for tokens in sentence:
            pwords = deque([""] * max_gram)
            for token in tokens:
                word, offset = token[0].text, token[2].text
                pwords.popleft()
                pwords.append(word)

                # deal with new sections
                if cp_idx < len(cp_markers) - 1 and int(token[2].text) == cp_markers[cp_idx + 1]:
                    cp_idx += 1
                    cp_counts.append({})
                    cp_dict = cp_counts[-1]

                # get candidates from pwords and add to appropriate dicts
                word_list = []
                biggest = None
                for i in range(1, len(pwords)+1):
                    word_list.insert(0, pwords[-i])
                    word_tuple = tuple(word_list)
                    if word_tuple in ngrams:
                        biggest = word_tuple

                if biggest is not None:
                    if biggest in cp_dict:
                        cp_dict[biggest] += 1
                    else:
                        cp_dict[biggest] = 1

    for cand in ngrams:
        count = 0
        for cp_count in cp_counts:
            if cand in cp_count:
                count += cp_count[cand]
        ngrams[cand] = count
    return cp_counts

def get_tag_features(tree, ngrams, pairs):
    """Extract NER, POS, and capitalization-based features for candidates

    Go through mapping of candidate n-grams to features and extract tag-based
    features (NER, POS, capitalization) using parsed Stanford NLP tree. The
    new features are added directly to the current feature mappings.

    Features:
        avg_ner (float): The average token fraction with PERSON or MISC NER tag
        avg_last_ner (float): The last token fraction with PERSON/MISC NER tag
        avg_cap (float): The average token fraction with first letter cap
        avg_last_cap (float): The last token fraction with first letter cap

    Args:
        tree (ElementTree): The parsed Stanford NLP tree.
            This tree should be made with annotations tokenize, ssplit, pos,
            lemma, and ner.
        ngrams (dict[tuple[str] -> dict]): The candidate to feature mapping.
            Candidates should be represented as string token tuples and
            features should be in a dict mapping string feature names to
            values.

    Returns:
        Nothing

        The method only changes the passed candidate feature dictionaries.
    """

    ner_feats = set(['avg_ner', 'avg_last_ner', 'avg_cap', 'avg_last_cap'])
    # get max gram length for tracking, initialize tag features
    max_gram = max(map(lambda x: len(x), ngrams.keys()))
    for ngram in ngrams:
        features = ngrams[ngram]
        for feat in ner_feats:
            features[feat] = 0

    for cand in ngrams:
        feats = ngrams[cand]
        lcaps = 1 if any(l.isupper() for l in cand[-1]) else 0
        num_caps = 0
        for token in cand:
            if any(l.isupper() for l in token):
                num_caps += 1
        feats['avg_last_cap'] = lcaps * 1.0
        feats['avg_cap'] = num_caps * 1.0

    # loop through Stanford NLP tree, checking tags when candidates appear
    counter, nercounter = 0, 0
    root = tree.getroot()
    sentences = root[0][0]
    for sentence in sentences:
        for tokens in sentence:
            # track previous words, NER tags, and capitalization
            pwords = deque([""] * max_gram)
            pner = deque([0] * max_gram)
            for token in tokens:
                word = token[0].text
                ner = 1 if (token[5].text == "MISC" or token[5].text == "PERSON") else 0
                word_list, ner_list = [], []

                pwords.popleft()
                pner.popleft()
                pwords.append(word)
                pner.append(ner)

                # go through candidates ending with current token
                biggest, big_ner = None, None
                for i in range(1, len(pwords)+1):
                    word_list.insert(0, pwords[-i])
                    ner_list.insert(0, pner[-i])
                    word_tuple = tuple(word_list)
                    ner_tuple = tuple(ner_list)
                    if word_tuple in ngrams:
                        biggest = word_tuple
                        big_ner = ner_tuple

                if biggest is not None:
                    features = ngrams[biggest]
                    features['avg_ner'] += ((sum(big_ner) * 1.0 / len(big_ner)) / features['count'])
                    features['avg_last_ner'] += (ner * 1.0 / features['count'])

    for pair in pairs:
        pair_feats = pairs[pair]
        cand1_feats = ngrams[pair[0]]
        cand2_feats = ngrams[pair[1]]
        for feat in ner_feats:
            pair_feats["1_" + feat] = cand1_feats[feat]
            pair_feats["2_" + feat] = cand2_feats[feat]

def get_tag_char_features(tree, ngrams):
    print 'Getting character tag features...'
    ner_feats = set(['avg_ner', 'avg_last_ner', 'avg_cap', 'avg_last_cap'])
    # get max gram length for tracking, initialize tag features
    max_gram = max(map(lambda x: len(x), ngrams.keys()))
    for ngram in ngrams:
        features = ngrams[ngram]
        for feat in ner_feats:
            features[feat] = 0

    for cand in ngrams:
        feats = ngrams[cand]
        lcaps = 1 if any(l.isupper() for l in cand[-1]) else 0
        num_caps = 0
        for token in cand:
            if any(l.isupper() for l in token):
                num_caps += 1
        feats['avg_last_cap'] = lcaps * 1.0
        feats['avg_cap'] = num_caps * 1.0

    # loop through Stanford NLP tree, checking tags when candidates appear
    counter, nercounter = 0, 0
    root = tree.getroot()
    sentences = root[0][0]
    for sentence in sentences:
        for tokens in sentence:
            # track previous words, NER tags, and capitalization
            pwords = deque([""] * max_gram)
            pner = deque([0] * max_gram)
            for token in tokens:
                word = token[0].text
                ner = 1 if (token[5].text == "MISC" or token[5].text == "PERSON") else 0
                word_list, ner_list = [], []

                pwords.popleft()
                pner.popleft()
                pwords.append(word)
                pner.append(ner)

                # go through candidates ending with current token
                biggest, big_ner = None, None
                for i in range(1, len(pwords)+1):
                    word_list.insert(0, pwords[-i])
                    ner_list.insert(0, pner[-i])
                    word_tuple = tuple(word_list)
                    ner_tuple = tuple(ner_list)
                    if word_tuple in ngrams:
                        biggest = word_tuple
                        big_ner = ner_tuple

                if biggest is not None:
                    features = ngrams[biggest]
                    features['avg_ner'] += ((sum(big_ner) * 1.0 / len(big_ner)) / features['count'])
    print 'Got character tag features...'

def get_tag_pair_features(pairs):
    ner_feats = set(['avg_ner', 'avg_last_ner', 'avg_cap', 'avg_last_cap'])

    for pair in pairs:
        pair_feats = pairs[pair]
        cand1_feats = ngrams[pair[0]]
        cand2_feats = ngrams[pair[1]]
        for feat in ner_feats:
            pair_feats["1_" + feat] = cand1_feats[feat]
            pair_feats["2_" + feat] = cand2_feats[feat]

def get_coref_features(ngrams, pairs):
    """Extract coreference features for candidates

    There may be multiple candidates which refer to the same character. This
    method attempts to account for this issue by adding features which reflect
    the other references for a given candidate: namely whether there are any
    and their occurence frequencies.

    Features:
        coref_shorter (bool (1/0)): whether there's a shorter coreference or not
        coref_longer (bool (1/0)): whether there's a longer coreference or not
        coref_shorter_count (int): the total frequency of the shorter coreferences
        coref_longer_count (int): the total frequency of the longer coreferences

    Args:
        ngrams (dict[tuple[str]->dict]): the mapping from candidates to features
            Candidates should be represented as string token tuples and
        features should be in a dict mapping string feature names to
        values.

    Returns:
        Nothing.

        The method only changes the passed candidate feature dictionaries.
    """
    section_types = ["st", "pg", "cp"]
    coref_feats = set(["coref_shorter", "coref_longer"])

    full_count_feats = set([
        'count', 'count_norm_length', 'count_norm_char'
    ])

    section_count_feats = set([
        'count', 'count_norm_length', 'count_norm_char',
        'cooc_cand', 'cooc_cand_norm_sec', 'cooc_cand_norm_char',
        'cooc_sec', 'cooc_sec_norm_length', 'cooc_cand_dd_norm_sec',
        'cooc_cand_dd_norm_char'
    ])

    # initialize relevant coreference features
    for ngram in ngrams:
        features = ngrams[ngram]
        features["coref_shorter"] = 0
        features["coref_longer"] = 0
        for feature in full_count_feats:
            features["coref_shorter_" + feature] = 0
            features["coref_longer_" + feature] = 0
        for section_type in section_types:
            for feature in section_count_feats:
                features["coref_shorter_" + feature + "_" + section_type] = 0
                features["coref_longer_" + feature + "_" + section_type] = 0

    coref_graph = dbg.disambiguate(ngrams)
    for candidate in coref_graph:
        features = ngrams[candidate]
        long_matches = coref_graph[candidate]
        if len(long_matches) > 0:
            features["coref_longer"] = 1
        for match in long_matches:
            match_feats = ngrams[match]
            match_feats["coref_shorter"] = 1
            for feature in full_count_feats:
                features["coref_longer_" + feature] += match_feats[feature]
                match_feats["coref_shorter_" + feature] += features[feature]
            for section_type in section_types:
                for feature in section_count_feats:
                    featstring = feature + "_" + section_type
                    features["coref_longer_" + featstring] += match_feats[featstring]
                    match_feats["coref_shorter_" + featstring] += features[featstring]

    for pair in pairs:
        pair_feats = pairs[pair]
        cand1_feats = ngrams[pair[0]]
        cand2_feats = ngrams[pair[1]]
        for feat in coref_feats:
            pair_feats["1_" + feat] = cand1_feats[feat]
            pair_feats["2_" + feat] = cand2_feats[feat]
        for feat in full_count_feats:
            for prefeat in coref_feats:
                fullfeat = (prefeat + "_" + feat)
                pair_feats["1_" + fullfeat] = cand1_feats[fullfeat]
                pair_feats["2_" + fullfeat] = cand2_feats[fullfeat]
        for feat in section_count_feats:
            for prefeat in coref_feats:
                for section_type in section_types:
                    fullfeat = (prefeat + "_" + feat + "_" + section_type)
                    pair_feats["1_" + fullfeat] = cand1_feats[fullfeat]
                    pair_feats["2_" + fullfeat] = cand2_feats[fullfeat]

def get_coref_char_features(ngrams):
    print 'Getting character coref features...'
    section_types = ["st", "pg", "cp"]
    coref_feats = set(["coref_shorter", "coref_longer"])

    full_count_feats = set([
        'count', 'count_norm_length', 'count_norm_char'
    ])

    section_count_feats = set([
        'count', 'count_norm_length', 'count_norm_char',
        'cooc_cand', 'cooc_cand_norm_sec', 'cooc_cand_norm_char',
        'cooc_sec', 'cooc_sec_norm_length', 'cooc_cand_dd_norm_sec',
        'cooc_cand_dd_norm_char'
    ])

    # initialize relevant coreference features
    for ngram in ngrams:
        features = ngrams[ngram]
        features["coref_shorter"] = 0
        features["coref_longer"] = 0
        for feature in full_count_feats:
            features["coref_shorter_" + feature] = 0
            features["coref_longer_" + feature] = 0
        for section_type in section_types:
            for feature in section_count_feats:
                features["coref_shorter_" + feature + "_" + section_type] = 0
                features["coref_longer_" + feature + "_" + section_type] = 0

    coref_graph = dbg.disambiguate(ngrams)
    for candidate in coref_graph:
        features = ngrams[candidate]
        long_matches = coref_graph[candidate]
        if len(long_matches) > 0:
            features["coref_longer"] = 1
        for match in long_matches:
            match_feats = ngrams[match]
            match_feats["coref_shorter"] = 1
            for feature in full_count_feats:
                features["coref_longer_" + feature] += match_feats[feature]
                match_feats["coref_shorter_" + feature] += features[feature]
            for section_type in section_types:
                for feature in section_count_feats:
                    featstring = feature + "_" + section_type
                    features["coref_longer_" + featstring] += match_feats[featstring]
                    match_feats["coref_shorter_" + featstring] += features[featstring]
    print 'Got character coref features!'

def get_coref_pair_features(pairs):
    section_types = ["st", "pg", "cp"]
    coref_feats = set("coref_shorter", "coref_longer")

    full_count_feats = set([
        'count', 'count_norm_length', 'count_norm_char'
    ])

    section_count_feats = set([
        'count', 'count_norm_length', 'count_norm_char',
        'cooc_cand', 'cooc_cand_norm_sec', 'cooc_cand_norm_char',
        'cooc_sec', 'cooc_sec_norm_length', 'cooc_cand_dd_norm_sec',
        'cooc_cand_dd_norm_char'
    ])

    for pair in pairs:
        pair_feats = pairs[pair]
        cand1_feats = ngrams[pair[0]]
        cand2_feats = ngrams[pair[1]]
        for feat in coref_feats:
            pair_feats["1_" + feat] = cand1_feats[feat]
            pair_feats["2_" + feat] = cand2_feats[feat]
        for feat in full_count_feats:
            for prefeat in coref_feats:
                fullfeat = (prefeat + "_" + feat)
                pair_feats["1_" + fullfeat] = cand1_feats[fullfeat]
                pair_feats["2_" + fullfeat] = cand2_feats[fullfeat]
        for feat in section_count_feats:
            for prefeat in coref_feats:
                for section_type in section_types:
                    fullfeat = (prefeat + "_" + feat + "_" + section_type)
                    pair_feats["1_" + fullfeat] = cand1_feats[fullfeat]
                    pair_feats["2_" + fullfeat] = cand2_feats[fullfeat]

def get_count_features(tree, markers, ngrams):
    """Extract candidate frequency and co-occurrence features.

    Using the parsed Stanford NLP tree, get sentence, paragraph, and chapter
    frequencies for the candidates, as well as co-occurrences of candidates
    in sentences, paragraphs, and chapters.

    Features:
        count_st: the number of sentences the candidate appears in
        count_pg: the number of paragraphs the candidate appears in
        count_cp: the number of chapters the candidate appears in
        cooc_st: the number of sentences the candidate co-occurs with others
        cooc_pg: the number of paragraphs the candidate co-occurs with others
        cooc_cp: the number of chapters the candidate co-occurs with others

    Args:
        tree (ElementTree): the parsed Stanford NLP tree
            This tree should be made with annotations tokenize, ssplit, pos,
                lemma, and ner.
            ngrams (dict[tuple[str]->dict):
                Candidates should be represented as string token tuples and
                features should be in a dict mapping string feature names to
                values.
            markers (dict[str->list[int]]):
                The dictionary maps section types to lists of character offsets
                from the beginning of the raw book text file. The character offsets
                indicating beginnings of new sections.

    Returns:
        Nothing.

        The method only changes the passed candidate feature dictionaries.
    """

    print 'Getting count features...'

    # track sentences, paragraphs, and chapters
    section_idx = {'sentence': -1, 'paragraph': -1, 'chapter': -1}
    section_counts = {'sentence': [], 'paragraph': [], 'chapter': []}
    section_markers = markers

    # current dictionaries counting candidate occurrences
    section_dicts = {'sentence': {}, 'paragraph': {}, 'chapter': {}}
    section_types = section_dicts.keys()

    count_feats = set([
        'count',
        'count_norm_length',
        'count_norm_char',
        'cooc_cand',
        'cooc_cand_norm_sec',
        'cooc_cand_norm_char',
        'cooc_sec',
        'cooc_sec_norm_length',
        'cooc_cand_dd_norm_sec',
        'cooc_cand_dd_norm_char'
    ])
    max_gram = max(map(lambda x: len(x), ngrams.keys()))

    # loop through tree and count candidates in sections
    root = tree.getroot()
    sentences = root[0][0]
    for sentence in sentences:
        for tokens in sentence:
            pwords = deque([""] * max_gram)
            for token in tokens:
                word, offset = token[0].text, token[2].text
                pwords.popleft()
                pwords.append(word)

                # deal with new sections
                for section_type in section_types:
                    idx = section_idx[section_type]
                    counts = section_counts[section_type]
                    marks = section_markers[section_type]
                    if idx < len(marks) - 1 and int(token[2].text) == marks[idx + 1]:
                        section_idx[section_type] += 1
                        counts.append({})
                        section_dicts[section_type] = counts[-1]

                # get candidates from pwords and add to appropriate dicts
                word_list = []
                biggest = None
                for i in range(1, len(pwords)+1):
                    word_list.insert(0, pwords[-i])
                    word_tuple = tuple(word_list)
                    if word_tuple in ngrams:
                        biggest = word_tuple

                if biggest is not None:
                    for section_type in section_types:
                        section_dict = section_dicts[section_type]
                        if biggest in section_dict:
                            section_dict[biggest] += 1
                        else:
                            section_dict[biggest] = 1

    # convert sentence, paragraph, chapter count dicts into matrices (sparse)
    for section_type in section_types:
        counts = section_counts[section_type]
        vectorizer = DictVectorizer(sparse=True)
        section_mat = vectorizer.fit_transform(counts)
        index = {v: k for k, v in vectorizer.vocabulary_.items()}
        marg_mat_full_norm = [ngrams[index[idx]]['count_norm_char'] for idx in range(len(index.keys()))]
        marg_mat_full_norm = np.array(marg_mat_full_norm)
        best_ind = None
        if marg_mat_full_norm.shape[0] < 40:
            best_ind = np.array(range(marg_mat_full_norm.shape[0]))
        else:
            best_ind = np.argpartition(marg_mat_full_norm, -40)[-40:]
        major_filter = np.zeros(marg_mat_full_norm.shape[0])
        major_filter[best_ind] = 1
        major_filter_sparse = lil_matrix((major_filter.shape[0], major_filter.shape[0]))
        major_filter_sparse.setdiag(major_filter)

        # marginalization to get total sentence, paragraph, chapter frequencies
        uform_mat = section_mat.copy()
        uform_mat[uform_mat > 0] = 1.0
        marg_mat = uform_mat.sum(axis=0)

        # section count and character over section count normalization
        marg_mat_len_norm = marg_mat / len(section_markers[section_type])
        marg_mat_count_norm = marg_mat / marg_mat.sum()

        # number of sections co-occurred in (as opposed to num co-occurrences in section)
        uform_major = uform_mat.dot(major_filter_sparse)
        uform_major = uform_major.sum(axis=1)
        uform_major[uform_major >= 1] = 1
        cooc_sec = (uform_mat.T).dot(uform_major)
        marg_cooc_sec_len_norm = cooc_sec / len(section_markers[section_type])

        # matrix multiplication to get co-occurrence sentence, paragraph, chapter matrices
        # TODO: might want to store these on disk since computation is expensive
        cooc_mat = (uform_mat.T).dot(uform_mat)
        cooc_mat_pre_norm = normalize(cooc_mat, norm='l1', axis=1)
        marg_sparse_count_norm = lil_matrix(cooc_mat.shape)
        marg_sparse_full_norm = lil_matrix(cooc_mat.shape)
        marg_sparse_count_norm.setdiag(marg_mat_count_norm.A1)
        marg_sparse_full_norm.setdiag(marg_mat_full_norm)
        cooc_mat_count_norm = cooc_mat_pre_norm.dot(marg_sparse_count_norm)
        cooc_mat_full_norm = cooc_mat_pre_norm.dot(marg_sparse_full_norm)
        cooc_uform = cooc_mat.copy()
        cooc_uform[cooc_uform > 0] = 1.0
        cooc_uform_count_norm = cooc_uform.dot(marg_sparse_count_norm)
        cooc_uform_full_norm = cooc_uform.dot(marg_sparse_full_norm)

        # marginalization to get total sentence, paragraph, chapter co-occurrences for each candidate
        marg_cooc = cooc_mat.sum(axis=1)
        marg_cooc_count_norm = cooc_mat_count_norm.sum(axis=1)
        marg_cooc_full_norm = cooc_mat_full_norm.sum(axis=1)
        marg_cooc_uform = cooc_uform.sum(axis=1)
        marg_cooc_uform_count_norm = cooc_uform_count_norm.sum(axis=1)
        marg_cooc_uform_full_norm = cooc_uform_full_norm.sum(axis=1)

        for idx in index:
            ngram, count, cooc = index[idx], marg_mat[0, idx], marg_cooc[idx, 0]
            count_norm_length = marg_mat_len_norm[0, idx]
            count_norm_char = marg_mat_count_norm[0, idx]
            cooc_sec_val = cooc_sec[idx, 0]
            cooc_sec_norm_length = marg_cooc_sec_len_norm[idx, 0]
            cooc_total_char_norm_sec = marg_cooc_count_norm[idx, 0]
            cooc_total_char_norm_char = marg_cooc_full_norm[idx, 0]
            cooc_char_norm_sec = marg_cooc_uform_count_norm[idx, 0]
            cooc_char_norm_char = marg_cooc_uform_full_norm[idx, 0]

            features = ngrams[ngram]
            features["count_" + abbrv(section_type)] = count
            features["count_norm_length_" + abbrv(section_type)] = count_norm_length
            features["count_norm_char_" + abbrv(section_type)] = count_norm_char
            features["cooc_cand_" + abbrv(section_type)] = cooc
            features["cooc_cand_norm_sec_" + abbrv(section_type)] = cooc_total_char_norm_sec
            features["cooc_cand_norm_char_" + abbrv(section_type)] = cooc_total_char_norm_char
            features["cooc_sec_" + abbrv(section_type)] = cooc_sec_val
            features["cooc_sec_norm_length_" + abbrv(section_type)] = cooc_sec_norm_length
            features["cooc_cand_dd_norm_sec_" + abbrv(section_type)] = cooc_char_norm_sec
            features["cooc_cand_dd_norm_char_" + abbrv(section_type)] = cooc_char_norm_char
            '''
            for idx2 in index:
                if idx == idx2:
                    continue
                pair = (ngram, index[idx2])
                pair_feats = pairs[pair]
                pair_feats["cooc_" + abbrv(section_type)] = cooc_mat[idx, idx2]
                pair_feats["cooc_norm_sec_" + abbrv(section_type)] = cooc_mat_count_norm[idx, idx2]
                pair_feats["cooc_norm_char_" + abbrv(section_type)] = cooc_mat_full_norm[idx, idx2]
                pair_feats["cooc_bool_" + abbrv(section_type)] = cooc_uform[idx, idx2]
                pair_feats["cooc_bool_norm_sec_" + abbrv(section_type)] = cooc_uform_count_norm[idx, idx2]
                pair_feats["cooc_bool_norm_char_" + abbrv(section_type)] = cooc_uform_full_norm[idx, idx2]
            '''

        '''
        for pair in pairs:
            pair_feats = pairs[pair]
            cand1_feats = ngrams[pair[0]]
            cand2_feats = ngrams[pair[1]]
            for feat in count_feats:
                sec_feat = feat + "_" + abbrv(section_type)
                pair_feats["1_" + sec_feat] = cand1_feats[sec_feat]
                pair_feats["2_" + sec_feat] = cand2_feats[sec_feat]
        '''
    print 'Got count features!'

def get_char_features(tokenfile, nlpfile, cutoffs, cp_cutoff):
    tree = ET.parse(nlpfile)
    markers = section(tree, tokenfile)
    candidates = get_candidates(tree, markers, cutoffs, cp_cutoff)
    get_count_features(tree, markers, candidates)
    get_tag_char_features(tree, candidates)
    get_coref_char_features(candidates)
    return candidates

def output(candidates):
    gram_size = 1
    while True:
        gram_candidates = { k: v['count'] for k, v in candidates.items() if len(k) == gram_size }
        sorted_grams = sorted(gram_candidates.items(), key=operator.itemgetter(1))
        if len(sorted_grams) == 0:
            break
        for i in range(len(sorted_grams)):
            key = sorted_grams[i][0]
            features = candidates[key]
            print "{0}: {1}".format(key, features)
        gram_size += 1

def write_char_feature_file(raw_fname, outdir, ngrams):
    raw_text_name = raw_fname.split('/')[-1]
    char_name_split = raw_text_name.split('.')
    char_name_split[0] += '_char_features'
    outfname = '.'.join(char_name_split)
    wfile = open(outdir + '/' + outfname, 'w')
    wfile.write(str(ngrams))
    wfile.close()

def write_pair_feature_file(raw_fname, outdir, pairs):
    pair_name_split = raw_text_name.split('.')
    pair_name_split[0] += '_pair_features'
    pairfname = '.'.join(pair_name_split)
    wfile = open(outdir + '/' + pairfname, 'w')
    wfile.write(str(pairs))
    wfile.close()

def write_readable_char_feature_file(raw_fname, outdir, ngrams):
    raw_text_name = raw_fname.split('/')[-1]
    char_name_split = raw_text_name.split('.')
    char_name_split[0] += '_char_features_readable'
    outfname = '.'.join(char_name_split)

    maxgram = max([len(ngram) for ngram in ngrams])
    filestr = []
    for size in range(1, maxgram+1):
        grams = [ngram for ngram in ngrams.keys() if len(ngram) == size]
        for gram in grams:
            features = ngrams[gram]
            filestr.append(str(gram))
            filestr.append('\n')
            for feature in features:
                filestr.append('\t')
                filestr.append(feature + ':' + str(features[feature]))
                filestr.append('\n')
    wfile = open(outdir + '/' + outfname, 'w')
    wfile.write(''.join(filestr))
    wfile.close()

def write_readable_pair_feature_file(raw_fname, outdir, pairs):
    pair_name_split = raw_text_name.split('.')
    pair_name_split[0] += '_pair_features_readable'
    pairfname = '.'.join(pair_name_split)
    filestr = []
    for size in range(1, maxgram+1):
        for psize in range(1, maxgram+1):
            opairs = [pair for pair in pairs.keys() if len(pair[0]) == size and len(pair[1]) == psize]
            for pair in opairs:
                features = pairs[pair]
                filestr.append(str(pair))
                filestr.append('\n')
                for feature in features:
                    filestr.append('\t')
                    filestr.append(feature + ':' + str(features[feature]))
                    filestr.append('\n')
    wfile = open(outdir + '/' + pairfname, 'w')
    wfile.write(''.join(filestr))
    wfile.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract candidate characters and feature values")
    parser.add_argument('-f', '--file', nargs=1, required=True, help='Book coreNLP file to extract candidates from')
    parser.add_argument('-rf', '--tokenfile', nargs=1, required=True, help='Book tokens file to extract candidates from')
    parser.add_argument('-o', '--outdir', nargs=1, required=True, help='Output directory for feature dict')
    parser.add_argument('-dr', '--fulldir', required=False, default=False, action='store_true', help='Run extraction for full directory')
    parser.add_argument('-d', '--debug', required=False, default=False, action='store_true', help='Whether to print output at all feature extraction steps')
    parser.add_argument('-n', '--numcands', nargs=1, required=False, default='flex', help='number of 1gram, 2gram, 3gram candidates')
    parser.add_argument('-cn', '--numcpcands', nargs=1, required=False, default='flex', help='number of tested chapter candidates')

    args = vars(parser.parse_args())
    debug = args['debug']
    tokens_text = args['tokenfile'][0]
    outdir = args['outdir'][0]
    full = args['fulldir']
    nlp_file = args['file'][0]
    cutoffs = args['numcands'][0]
    cutoffs = 'flex' if cutoffs == 'f' else eval(cutoffs)
    cp_cutoff = args['numcpcands'][0]
    cp_cutoff = 'flex' if cp_cutoff == 'f' else eval(cp_cutoff)

    if not full:
        # test candidate selection

        candidates = get_char_features(tokens_text, nlp_file, cutoffs, cp_cutoff)
        write_char_feature_file(tokens_text, outdir, candidates)
        write_readable_char_feature_file(tokens_text, outdir, candidates)
    else:
        all_tokens = os.listdir(tokens_text)
        all_nlp = [nlp_file + '/' + f + '.xml' for f in all_tokens]
        all_tokens = [tokens_text + '/' + f for f in all_tokens]
        for i in range(len(all_tokens)):
            tokens, nlp = all_tokens[i], all_nlp[i]
            print "Starting {0}".format(tokens.split('/')[-1])
            try:
                candidates = get_char_features(tokens, nlp, cutoffs, cp_cutoff)
                write_char_feature_file(tokens, outdir, candidates)
                write_readable_char_feature_file(tokens, outdir, candidates)
                print "Feature Parsing for {0}: SUCCESS".format(tokens.split('/')[-1])
            except Exception as e:
                traceback.print_exc()
                print "Feature Parsing for {0}: FAILURE".format(tokens.split('/')[-1])
                continue
