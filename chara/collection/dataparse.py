import os
import xml.etree.ElementTree as ET
from django.utils.encoding import smart_str
import utils.wordnet_hyponyms as wh
from nltk.corpus import wordnet as wn
from subprocess import call

def parse_corenlp(book_dir='raw', out_dir='./'):
    # get all books in book full text dir
    all_books = os.listdir(book_dir)
    all_books = [book_dir + '/' + book for book in all_books]

    # create book name index for batch CoreNLP processing
    book_list = open(book_dir + '/book_list.txt', 'w')
    book_list.write('\n'.join(all_books))
    book_list.close()

    nlp_dir = os.environ['CORE_NLP']
    if not os.path.exists(out_dir + '/corenlp'):
        os.makedirs(out_dir + '/corenlp')
    out_nlpdir = out_dir + '/corenlp'
    out_tokensdir = out_dir + '/tokens'

    curr_dir = os.path.dirname(os.path.realpath(__file__))
    call(["bash", nlp_dir + "/corenlp.sh", "--annotators", "tokenize,ssplit,pos,lemma,ner", "-filelist", book_dir + "/book_list.txt", "-outputDirectory", out_nlpdir])
    call(["bash", curr_dir + "/tokenize.sh", book_dir, out_tokensdir])

    os.remove(book_dir + '/book_list.txt')

def parse_candidates(nlp_dir='corenlp', tokens_dir='tokens', out_dir='./'):
    if not os.path.exists(out_dir + '/candidates'):
        os.makedirs(out_dir + '/candidates')
    fullout = out_dir + '/candidates'
    all_corenlp = os.listdir(nlp_dir)
    all_corenlp = sorted(all_corenlp)
    all_tokens = os.listdir(tokens_dir)
    all_tokens = sorted(all_tokens)
    all_full_corenlp = [nlp_dir + '/' + fname for fname in all_corenlp]
    all_full_tokens = [tokens_dir + '/' + fname for fname in all_tokens]
    for i in range(len(all_full_corenlp)):
        corenlp = all_full_corenlp[i]
        tokens = all_full_tokens[i]
        tree = ET.parse(corenlp)
        markers = section(tree, tokens)
        candidates = get_candidates(
            tree,
            markers,
            num_cutoffs=[100, 50, 50, 10, 10],
            num_cp_cutoff=5,
            per_cutoffs=[30, 40, 40, 10, 10],
            per_cp_cutoff=30
        )
        outfile = open(fullout + '/' + all_tokens[i], 'w')
        outfile.write(str(candidates))
        outfile.close()

def get_token_lines(token_fname):
    token_lines = open(token_fname, 'r')
    token_lines = token_lines.readlines()
    token_lines = [token[:-1] for token in token_lines]
    token_lines = token_lines[2:]
    return token_lines

def section(tree, token_fname):
    markers = {}
    all_sentences = tree.getroot()[0][0]
    token_lines = get_token_lines(token_fname)

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
                    idx_start, long_token_string = tl_idx, ''
                    while (smart_str(token_lines[tl_idx]) != token_t) and not token_t in long_token_string:
                        test_token = token_lines[tl_idx]
                        if test_token == '*NL*':
                            nl_count += 1
                        else:
                            long_token_string += test_token
                        tl_idx += 1
                    if token_lines[tl_idx] != token_t and token_t in long_token_string:
                        tl_idx -= 1
                except:
                    print "FAIL", token_filename
                    return
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
        if tl_idx != len(token_lines):
            print "FAIL", token_filename
    markers['sentence'] = st_markers
    markers['paragraph'] = pg_markers
    markers['chapter'] = cp_markers
    markers['book'] = st_markers[-1]

    return markers

def is_the(word):
    if word == 'the' or word == 'The' or word == 'THE':
        return True
    return False

def get_candidates(
        tree,
        markers,
        num_cutoffs='flex',
        num_cp_cutoff='flex',
        per_cutoffs='flex',
        per_cp_cutoff='flex',
        caps_only=False):

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
    marg_ngrams = {}
    for i in range(len(ngrams)):
        cp_ngrams = ngrams[i]
        for key in cp_ngrams:
            if key in marg_ngrams:
                marg_ngrams[key] += cp_ngrams[key]
            else:
                marg_ngrams[key] = cp_ngrams[key]

    dedup_candidates(tree, marg_ngrams)
    ngrams = get_general_count_feature(tree, marg_ngrams, markers)

    # get frequent candidates across whole book
    filtered_grams = []
    if per_cutoffs == 'flex':
        try_grams = (sorted(marg_ngrams.items(), key=operator.itemgetter(1)))
        try_grams = [gram for gram in try_grams if gram[1] > 2]
        if num_cutoffs != 'flex':
            for gram_size in range(1, 8):
                if gram_size <= len(num_cutoffs):
                    size_grams = [gram for gram in try_grams if len(gram[0])==i]
                    size_grams = sorted(size_grams, key=operator.itemgetter(1))
                    size_grams = size_grams[-num_cutoffs[gram_size-1]:]
                    filtered_grams.extend(size_grams)
    else:
        for gram_size in range(1, 8):
            grams = { gram: marg_ngrams[gram] for gram in marg_ngrams.keys() if len(gram) == gram_size }
            norm_grams = {}
            total = 0
            for gram in grams:
                total += grams[gram]
            for gram in grams:
                norm_grams[gram] = grams[gram] * 1.0 / total
            sorted_grams = sorted(grams.items(), key=operator.itemgetter(1))
            if gram_size <= len(per_cutoffs):
                if len(sorted_grams) == 0:continue
                cutoff_idx = int(-1.0 * per_cutoffs[gram_size-1] / 100 * len(sorted_grams))
                pass_grams = sorted_grams[cutoff_idx:]
                # don't include them if they don't occur often
                pass_grams = [gram for gram in pass_grams if gram[1] > 2]
                pass_grams.extend([(gram, grams[gram]) for gram in grams if norm_grams[gram] > 0.05])
                pass_grams = list(set(pass_grams))
                if num_cutoffs != 'flex':
                    if len(pass_grams) >= num_cutoffs[gram_size-1]:
                        pass_grams = sorted(pass_grams, key=operator.itemgetter(1))
                        pass_grams = pass_grams[-num_cutoffs[gram_size-1]:]
                filtered_grams.extend(pass_grams)
            else:
                # don't include them if they don't occur often
                pass_grams = [sorted_grams[idx] for idx in range(len(sorted_grams)) if sorted_grams[idx][1] > 3]
                pass_grams.extend([(gram, grams[gram]) for gram in grams if norm_grams[gram] > 0.05])
                pass_grams = list(set(pass_grams))
                if not num_cutoffs == 'flex':
                    if len(pass_grams) >= 3:
                        pass_grams = pass_grams[-3:]
                filtered_grams.extend(pass_grams)

    # get frequent candidates per chapter
    if per_cutoffs == 'flex':
        for cp_idx in range(len(ngrams)):
            cp_ngrams = ngrams[cp_idx]
            pass_grams = [(gram, marg_ngrams[gram]) for gram in cp_ngrams if cp_ngrams[gram] > 4]
            if num_cp_cutoff != 'flex':
                if len(pass_grams) >= num_cp_cutoff:
                    pass_grams = sorted(pass_grams, key=operator.itemgetter(1))
                    pass_grams = pass_grams[-num_cp_cutoff:]
            filtered_grams.extend(pass_grams)
    else:
        for cp_idx in range(len(ngrams)):
            cp_ngrams = ngrams[cp_idx]
            norm_grams = {}
            total = 0
            for gram in cp_ngrams:
                total += cp_ngrams[gram]
            for gram in cp_ngrams:
                norm_grams[gram] = cp_ngrams[gram] * 1.0 / total
            for cp_gram_size in range(1, 8):
                cp_grams = { gram: cp_ngrams[gram] for gram in cp_ngrams.keys() if len(gram) == cp_gram_size }
                sorted_cp_grams = sorted(cp_grams.items(), key=operator.itemgetter(1))
                if len(sorted_cp_grams) == 0: continue
                cutoff_idx = int(-1.0 * per_cp_cutoff / 100 * len(sorted_cp_grams))
                top_cp_grams = sorted_cp_grams[cutoff_idx:]
                pass_cp_grams = [(gram[0], marg_ngrams[gram[0]]) for gram in top_cp_grams if gram[1] > 3]
                pass_cp_grams.extend([(gram, cp_grams[gram]) for gram in cp_grams if norm_grams[gram] > 0.2])
                pass_cp_grams = list(set(pass_cp_grams))
                if num_cp_cutoff != 'flex':
                    if len(pass_cp_grams) >= num_cp_cutoff:
                        pass_cp_grams = sorted(pass_cp_grams, key=operator.itemgetter(1))
                        pass_cp_grams = pass_cp_grams[-num_cp_cutoff:]
                filtered_grams.extend([(gram, cp_grams[gram]) for gram in cp_grams if norm_grams[gram] > 0.2])

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
        candidates[key]['count_norm_length'] = candidates[key]['count'] * 1.0 / total_length
        candidates[key]['count_norm_char'] = candidates[key]['count'] * 1.0 / total

    print 'Got {0} candidates!'.format(len(candidates.keys()))
    return candidates

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
