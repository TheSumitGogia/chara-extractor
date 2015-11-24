import xml.etree.ElementTree as ET
import operator
import argparse
import itertools as it
from collections import deque
from sklearn.feature_extraction import DictVectorizer

SECTION_MAP = {'sentence': 'st', 'paragraph': 'pg', 'chapter': 'cp'}
def abbrv(section_name):
    return SECTION_MAP[section_name]

def section(tree, raw_filename):
    markers = {}
    def section_paragraphs_and_chapters(raw_fname):
        raw_file = open(raw_fname, 'r')
        raw_lines = raw_file.readlines()
        pg_markers = []
        cp_markers = []
        skip = 2
        char_offset = 0
        for i in range(len(raw_lines)):
            line = raw_lines[i]
            if line == '\n':
                skip += 1
                char_offset += 1
            elif skip >= 1:
                line_strip = line.lstrip()
                diff = len(line) - len(line_strip)
                if skip >= 2:
                    cp_markers.append(char_offset + diff)
                pg_markers.append(char_offset + diff)
                skip = 0
                char_offset += len(line)
            else:
                char_offset += len(line)
        markers['paragraph'] = pg_markers
        markers['chapter'] = cp_markers
        raw_file.close()
    def section_sentences(nlptree):
        root = nlptree.getroot()
        st_markers = []
        for document in root:
            for sentences in document:
                for sentence in sentence:
                    first_token = sentence[0][0]
                    offset = first_token[2]
                    st_markers.append(offset)
        markers['sentence'] = st_markers

    section_sentences(tree)
    section_paragraphs_and_chapters(raw_filename)

    return markers

# TODO: this method can be shortened easily, and should be
def get_candidates(tree, markers):
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

    # largest number of candidates to consider for different ngram sizes
    cutoffs = [20, 20, 5]
    cp_cutoff = 10

    # store candidates with counts per chapter
    ngrams = []

    # chapter offsets and tracking
    cp_markers = markers['chapter']
    cp_index = -1

    # stuff to indicate nesting of invalidity of candidate noun phrases
    bad_token_set = set(['Chapter', 'CHAPTER', 'said', ',', "''", 'and', ';', '-RSB-', '-LSB-', '_', '--', '``', '.'])
    bad_np_tags = set(['CC', 'IN', 'TO', 'WDT', 'WP', 'WP$', 'WRB', 'UH', 'VB', 'VBD', 'VBP', 'VBZ', 'MD'])

    # loop through Stanford NLP tree, extracting initial candidate set
    root = tree.getroot()
    for document in root:
        for sentences in document:
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
                            cp_index += 1

                        # filter for candidates with last word capital noun
                        if noun and word[0].isupper():
                            # loop through previous words, adding to noun phrase
                            curr_idx = token_idx
                            word_list = [word]
                            first_tag = token[4].text
                            np_condition = True
                            exhausted = False
                            while np_condition and not exhausted:
                                # check if valid candidate and count
                                word_tuple = tuple(word_list)
                                if word_tuple[0] in bad_token_set:
                                    exhausted = True
                                    break
                                if first_tag.startswith('NN'):
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
                                if word_tuple[0] in bad_token_set:
                                    exhausted = True
                                    break
                                if first_tag.startswith('NN'):
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
    check_keys = norm_ngrams.keys()

    # get frequent candidates across whole book
    gram_size = 1
    more_grams = True
    filtered_grams = []
    while more_grams:
        grams = { gram: norm_ngrams[gram] for gram in norm_ngrams.keys() if len(gram) == gram_size }
        sorted_grams = sorted(grams.items(), key=operator.itemgetter(1))
        if gram_size <= 3:
            pass_grams = sorted_grams[-1 * cutoffs[gram_size - 1]:]
            # don't include them if they don't occur often
            pass_grams = [gram for gram in pass_grams if gram[1] > 2]
            filtered_grams.extend(pass_grams)
            gram_size += 1
        else:
            # don't include them if they don't occur often
            pass_grams = [sorted_grams[idx] for idx in range(len(sorted_grams)) if sorted_grams[idx][1] > 5]
            filtered_grams.extend(pass_grams)
            gram_size += 1
            # stop if no n-grams are being passed
            if len(pass_grams) == 0:
                more_grams = False

    # get frequenct candidates per chapter
    for cp_idx in range(len(ngrams)):
        cp_ngrams = ngrams[cp_idx]
        cp_gram_size = 1
        more_cp_grams = True
        while more_cp_grams:
            cp_grams = { gram: cp_ngrams[gram] for gram in cp_ngrams.keys() if len(gram) == cp_gram_size }
            sorted_cp_grams = sorted(cp_grams.items(), key=operator.itemgetter(1))
            top_cp_grams = sorted_cp_grams[-1 * cp_cutoff:]
            print 'top chapter {0} grams: {1}'.format(cp_idx, top_cp_grams)
            pass_cp_grams = [(gram[0], norm_ngrams[gram[0]]) for gram in top_cp_grams if gram[1] > 5]
            filtered_grams.extend(pass_cp_grams)
            cp_gram_size += 1
            if len(pass_cp_grams) == 0:
                more_cp_grams = False

    # changing list of candidate tuples with counts to feature map
    candidates = {}
    for i in range(len(filtered_grams)):
        key = filtered_grams[i][0]
        count = filtered_grams[i][1]
        candidates[key] = {'count': count}
        candidates[key]['length'] = len(key)

    return candidates

def get_tag_features(tree, ngrams):
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

    # get max gram length for tracking, initialize tag features
    max_gram = max(map(lambda x: len(x) if isinstance(x, tuple) else 1, ngrams.keys()))
    for ngram in ngrams:
        features = ngrams[ngram]
        features['avg_ner'] = 0
        features['avg_last_ner'] = 0
        features['avg_cap'] = 0
        features['avg_last_cap'] = 0

    # loop through Stanford NLP tree, checking tags when candidates appear
    root = tree.getroot()
    for document in root:
        for sentences in document:
            for sentence in sentences:
                for tokens in sentence:
                    # track previous words, NER tags, and capitalization
                    pwords = deque([""] * max_gram)
                    pner = deque([0] * max_gram)
                    pcaps = deque([0] * max_gram)
                    for token in tokens:
                        word = token[0].text
                        ner = 1 if token[5].text == "MISC" or token[5].text == "PERSON" else 0
                        caps = 1 if token[0].text[0].isupper() else 0
                        word_list, ner_list, caps_list = [], [], []

                        pwords.popleft()
                        pner.popleft()
                        pcaps.popleft()
                        pwords.append(word)
                        pner.append(ner)
                        pcaps.append(caps)

                        # go through candidates ending with current token
                        for i in range(len(pwords)+1):
                            word_list.insert(0, pwords[-i])
                            ner_list.insert(0, pner[-i])
                            caps_list.insert(0, pcaps[-i])
                            word_tuple = tuple(word_list)
                            if word_tuple in ngrams:
                                features = ngrams[word_tuple]
                                features['avg_ner'] += ((sum(ner_list) * 1.0 / len(ner_list)) / features['count'])
                                features['avg_last_ner'] += (ner * 1.0 / features['count'])
                                features['avg_cap'] += ((sum(caps_list) * 1.0 / len(caps_list)) / features['count'])
                                features['avg_last_cap'] += (caps * 1.0 / features['count'])

def get_coref_features(ngrams):
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

    # initialize relevant coreference features
    for ngram in ngrams:
        features = ngrams[ngram]
        features["coref_shorter"] = 0
        features["coref_longer"] = 0
        features["coref_shorter_count"] = 0
        features["coref_longer_count"] = 0

    # go through each of the ngrams, check for token subset ngrams
    # TODO: make this way smarter - not only token subset
    # TODO: deal with subset counts including superset counts?
    for ngram in ngrams:
        if isinstance(ngram, tuple) and len(ngram) > 1:
            cont_features = ngrams[ngram]
            for r in range(1, len(ngram)):
                subs = it.combinations(ngram, r)
                for sub in subs:
                    sub = sub if len(sub) > 1 else sub[0]
                    print sub
                    if sub in ngrams:
                        features = ngrams[sub]
                        features["coref_longer"] = 1
                        features["coref_longer_count"] += cont_features["count"]
                        if cont_features["coref_shorter"] == 0:
                            cont_features["coref_shorter"] = 1
                        cont_features["coref_shorter_count"] += features["count"]

def get_count_features(tree, ngrams, markers):
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

    # track sentences, paragraphs, and chapters
    section_idx = {'sentence': -1, 'paragraph': -1, 'chapter': -1}
    section_counts = {'sentence': {}, 'paragraph': {}, 'chapter': {}}
    section_markers = markers

    # current dictionaries counting candidate occurrences
    section_dicts = {'sentence': {}, 'paragraph': {}, 'chapter': {}}
    section_types = section_dicts.keys()

    max_gram = max(map(lambda x: len(x), ngrams.keys()))

    # loop through tree and count candidates in sections
    root = tree.getroot()
    for document in root:
        for sentences in document:
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
                            dic = section_dicts[section_type]
                            marks = section_markers[section_type]
                            if idx < len(marks) - 1 and int(token[2].text) == marks[mark + 1]:
                                section_idx[section_type] += 1
                                counts.append({})
                                dic = counts[section_idx[section_type]]

                        # get candidates from pwords and add to appropriate dicts
                        word_list = []
                        for i in range(len(pwords)+1):
                            word_list.insert(0, pwords[-i])
                            word_tuple = tuple(word_list)
                            if word_tuple in ngrams:
                                for section_type in section_types:
                                    section_dict = section_dicts[section_type]
                                    if word_tuple in section_dict:
                                        section_dict[word_tuple] += 1
                                    else:
                                        section_dict[word_tuple] = 1

    # convert sentence, paragraph, chapter count dicts into matrices (sparse)
    section_v = {'sentence': None, 'paragraph': None, 'chapter': None}
    section_mats = {'sentence': None, 'paragraph': None, 'chapter': None}
    section_uforms = {'sentence': None, 'paragraph': None, 'chapter': None}
    section_marg = {'sentence': None, 'paragraph': None, 'chapter': None}
    section_cooc = {'sentence': None, 'paragraph': None, 'chapter': None}
    section_marg_cooc = {'sentence': None, 'paragraph': None, 'chapter': None}
    for section_type in section_types:
        counts = section_counts[section_type]
        section_v[section_type] = DictVectorizer(sparse=True)
        vectorizer = section_v[section_type]
        section_mats[section_type] = vectorizer.fit_transform(counts)

        # marginalization to get total sentence, paragraph, chapter frequencies
        section_uforms[section_type] = section_mats[section_type].copy()
        uform_mat = section_uforms[section_type]
        uform_mat[uform_mat > 0] = 1.0
        section_marg[section_type] = uform_mat.sum(axis=0)
        marg_mat = section_marg[section_type]

        # matrix multiplication to get co-occurrence sentence, paragraph, chapter matrices
        # TODO: might want to store these on disk since computation is expensive
        section_cooc[section_type] = (uform_mat.T).dot(uform_mat)
        cooc_mat = section_cooc[section_type]

        # marginalization to get total sentence, paragraph, chapter co-occurrences for each candidate
        section_marg_cooc[section_type] = cooc_mat.sum(axis=1)
        marg_cooc = section_marg_cooc[section_type]

        index = {v: k for k, v in section_v[section_type].vocabulary_.items()}
        for idx in index:
            ngram, count, cooc = index[idx], marg_mat[0, idx], marg_cooc[idx, 0]
            features = ngrams[ngram]
            features["count_" + abbrv(section_type)] = count
            features["cooc_" + abbrv(section_type)] = cooc

def output(candidates):
    gram_size = 1
    while True:
        gram_candidates = { k: v for k, v in candidates.items() if len(k) == gram_size }
        sorted_grams = sorted(gram_candidates.items(), key=operator.itemgetter(1))
        if len(sorted_grams == 0):
            break
        for i in range(len(sorted_grams)):
            key = sorted_grams[i][0],
            features = candidates[key]
            print "{0}: {1}".format(key, features)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract candidate characters and feature values")
    parser.add_argument('-f', '--file', nargs=1, required=True, help='Book coreNLP file to extract candidates from')
    parser.add_argument('-rf', '--rawfile', nargs=1, required=True, help='Book raw text file to extract candidates from')

    args = vars(parser.parse_args())
    raw_text = args['rawfile'][0]
    markers = section(raw_text)
    tree = ET.parse(args['file'][0])

    print "".join(["-"] * 10 + ["CANDIDATES"] + ["-"] * 10)
    candidates = get_candidates(tree, markers)
    output(candidates)

    print "\n"
    print "".join(["-"] * 10 + ["TAGGING"] + ["-"] * 10)
    get_tag_features(tree, candidates)
    output(candidates)

    print "\n"
    print "".join(["-"] * 10 + ["CONTAINMENT"] + ["-"] * 10)
    get_coref_features(candidates)
    output(candidates)

    print "\n"
    print "".join(["-"] * 10 + ["COUNTING"] + ["-"] * 10)
    get_count_features(candidates)
    output(candidates)
