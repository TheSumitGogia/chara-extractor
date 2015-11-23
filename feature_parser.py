import xml.etree.ElementTree as ET
import operator
import argparse
import itertools as it
from collections import deque
from sklearn.feature_extraction import DictVectorizer

def section(raw_fname):
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
    markers = {'paragraph': pg_markers, 'chapter': cp_markers}
    raw_file.close()
    return markers

# TODO: this method can be shortened easily, and should be
def get_candidates(tree, markers):
    cutoff_u = 20
    cutoff_b = 20
    cutoff_t = 5

    ugrams = []
    bgrams = []
    tgrams = []

    cp_markers = markers['chapter']
    cp_index = -1

    root = tree.getroot()
    for document in root:
        for sentences in document:
            for sentence in sentences:
                for tokens in sentence:
                    pword = ""
                    ppword = ""
                    for token in tokens:
                        if cp_index < len(cp_markers) - 1 and int(token[2].text) == cp_markers[cp_index + 1]:
                            ugrams.append({})
                            bgrams.append({})
                            tgrams.append({})
                            cp_index += 1

                        word = token[0].text
                        noun = (token[4].text.startswith('NN'))

                        if noun:
                            if word in ugrams[-1]:
                                ugrams[-1][word] += 1
                            else:
                                ugrams[-1][word] = 1
                            if not pword == '':
                                if (pword, word) in bgrams[-1]:
                                    bgrams[-1][(pword, word)] += 1
                                else:
                                    bgrams[-1][(pword, word)] = 1
                            if not ppword == '':
                                if (ppword, pword, word) in tgrams[-1]:
                                    tgrams[-1][(ppword, pword, word)] += 1
                                else:
                                    tgrams[-1][(ppword, pword, word)] = 1
                        ppword = pword
                        pword = word

    norm_ugrams = {}
    for i in range(len(ugrams)):
        cp_ugrams = ugrams[i]
        for key in cp_ugrams:
            if key in norm_ugrams:
                norm_ugrams[key] += cp_ugrams[key]
            else:
                norm_ugrams[key] = cp_ugrams[key]
    norm_bgrams = {}
    for i in range(len(bgrams)):
        cp_bgrams = bgrams[i]
        for key in cp_bgrams:
            if key in norm_bgrams:
                norm_bgrams[key] += cp_bgrams[key]
            else:
                norm_bgrams[key] = cp_bgrams[key]
    norm_tgrams = {}
    for i in range(len(tgrams)):
        cp_tgrams = tgrams[i]
        for key in cp_tgrams:
            if key in norm_tgrams:
                norm_tgrams[key] += cp_tgrams[key]
            else:
                norm_tgrams[key] = cp_tgrams[key]

    filt_ugrams = { ugram: norm_ugrams[ugram] for ugram in norm_ugrams.keys() if ugram[0].isupper() }
    filt_bgrams = { bgram: norm_bgrams[bgram] for bgram in norm_bgrams.keys() if bgram[1][0].isupper() }
    filt_tgrams = { tgram: norm_tgrams[tgram] for tgram in norm_tgrams.keys() if tgram[2][0].isupper() }
    sorted_ugrams = sorted(filt_ugrams.items(), key=operator.itemgetter(1))
    sorted_bgrams = sorted(filt_bgrams.items(), key=operator.itemgetter(1))
    sorted_tgrams = sorted(filt_tgrams.items(), key=operator.itemgetter(1))
    filtered_ugrams = sorted_ugrams[-cutoff_u:]
    filtered_bgrams = sorted_bgrams[-cutoff_b:]
    filtered_tgrams = sorted_tgrams[-cutoff_t:]

    for cp_idx in range(len(ugrams)):
        cp_ugrams = ugrams[cp_idx]
        cp_bgrams = bgrams[cp_idx]
        cp_tgrams = tgrams[cp_idx]
        filt_cp_ugrams = { ugram: cp_ugrams[ugram] for ugram in cp_ugrams.keys() if ugram[0].isupper() }
        filt_cp_bgrams = { bgram: cp_bgrams[bgram] for bgram in cp_bgrams.keys() if bgram[1][0].isupper() }
        filt_cp_tgrams = { tgram: cp_tgrams[tgram] for tgram in cp_tgrams.keys() if tgram[2][0].isupper() }
        sorted_cp_ugrams = sorted(filt_cp_ugrams.items(), key=operator.itemgetter(1))
        sorted_cp_bgrams = sorted(filt_cp_bgrams.items(), key=operator.itemgetter(1))
        sorted_cp_tgrams = sorted(filt_cp_tgrams.items(), key=operator.itemgetter(1))
        top_cp_ugrams = sorted_cp_ugrams[-10:]
        top_cp_bgrams = sorted_cp_bgrams[-10:]
        top_cp_tgrams = sorted_cp_tgrams[-10:]
        for i in range(len(top_cp_ugrams)):
            key = top_cp_ugrams[i][0]
            count = norm_ugrams[key]
            if top_cp_ugrams[i][1] > 5:
                filtered_ugrams.append((key, count))
        for i in range(len(top_cp_bgrams)):
            key = top_cp_bgrams[i][0]
            count = norm_bgrams[key]
            if top_cp_bgrams[i][1] > 5:
                filtered_bgrams.append((key, count))
        for i in range(len(top_cp_tgrams)):
            key = top_cp_tgrams[i][0]
            count = norm_tgrams[key]
            if top_cp_tgrams[i][1] > 5:
                filtered_tgrams.append((key, count))

    candidates = {}
    for i in range(len(filtered_ugrams)):
        key = filtered_ugrams[i][0]
        count = filtered_ugrams[i][1]
        candidates[key] = {'count': count}
        candidates[key]['length'] = 1
    for i in range(len(filtered_bgrams)):
        key = filtered_bgrams[i][0]
        count = filtered_bgrams[i][1]
        candidates[key] = {'count': count}
        candidates[key]['length'] = 2
    for i in range(len(filtered_tgrams)):
        key = filtered_tgrams[i][0]
        count = filtered_tgrams[i][1]
        candidates[key] = {'count': count}
        candidates[key]['length'] = 3

    return candidates

def get_tag_features(tree, ngrams):
    # features
    # 1) average last ner percentage
    # 2) average all ner percentage
    # 3) average capitalization percentage

    # get max gram length for tracking
    max_gram = max(map(lambda x: len(x) if isinstance(x, tuple) else 1, ngrams.keys()))
    for ngram in ngrams:
        features = ngrams[ngram]
        features['avg_ner'] = 0
        features['avg_last_ner'] = 0
        features['avg_cap'] = 0
        features['avg_last_cap'] = 0

    root = tree.getroot()
    for document in root:
        for sentences in document:
            for sentence in sentences:
                for tokens in sentence:
                    pwords = deque([""] * (max_gram - 1))
                    pner = deque([0] * (max_gram - 1))
                    pcaps = deque([0] * (max_gram - 1))
                    for token in tokens:
                        word = token[0].text
                        word_list = [word]
                        ner = 1 if token[5].text == "MISC" or token[5].text == "PERSON" else 0
                        ner_list = [ner]
                        caps = 1 if token[0].text[0].isupper() else 0
                        caps_list = [caps]
                        if word in ngrams:
                            features = ngrams[word]
                            features['avg_ner'] += ner * 1.0 / features['count']
                            features['avg_last_ner'] += ner * 1.0 / features['count']
                            features['avg_cap'] += caps * 1.0 / features['count']
                            features['avg_last_cap'] += caps * 1.0 / features['count']
                        for i in range(1, len(pwords)+1):
                            word_list.insert(0, pwords[-i])
                            ner_list.insert(0, pner[-i])
                            caps_list.insert(0, pcaps[-i])
                            word_tuple = tuple(word_list)
                            if word_tuple in ngrams:
                                if word_tuple == (',', 'and', 'Jim'):
                                    print "gogogogo"
                                features = ngrams[word_tuple]
                                features['avg_ner'] += ((sum(ner_list) * 1.0 / len(ner_list)) / features['count'])
                                features['avg_last_ner'] += (ner * 1.0 / features['count'])
                                features['avg_cap'] += ((sum(caps_list) * 1.0 / len(caps_list)) / features['count'])
                                features['avg_last_cap'] += (caps * 1.0 / features['count'])
                        pwords.popleft()
                        pner.popleft()
                        pcaps.popleft()
                        pwords.append(word)
                        pner.append(ner)
                        pcaps.append(caps)

def get_containment_features(ngrams):
    # features
    # 1) contained in another
    # 2) contains another
    for ngram in ngrams:
        features = ngrams[ngram]
        features["contained"] = 0
        features["contains"] = 0
        features["container_count"] = 0
        features["contains_count"] = 0

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
                        features["contained"] = 1
                        features["container_count"] += cont_features["count"]
                        if cont_features["contains"] == 0:
                            cont_features["contains"] = 1
                        cont_features["contains_count"] += features["count"]

def get_count_features(tree, ngrams, markers):
    sentence_idx = -1
    sentence_counts = []
    chapter_idx = -1
    chapter_counts = []
    paragraph_idx = -1
    paragraph_counts = []

    sentence_dict = {}
    paragraph_dict = {}
    chapter_dict = {}

    paragraph_markers = markers['paragraph']
    chapter_markers = markers['chapter']

    max_gram = max(map(lambda x: len(x) if isinstance(x, tuple) else 1, ngrams.keys()))
    root = tree.getroot()
    for document in root:
        for sentences in document:
            for sentence in sentences:
                sentence_idx += 1
                sentence_counts.append({})
                sentence_dict = sentence_counts[sentence_idx]
                for tokens in sentence:
                    pwords = deque([""] * (max_gram - 1))
                    for token in tokens:
                        if paragraph_idx < len(paragraph_markers) - 1 and int(token[2].text) == paragraph_markers[paragraph_idx + 1]:
                            paragraph_idx += 1
                            paragraph_counts.append({})
                            paragraph_dict = paragraph_counts[paragraph_idx]
                        if chapter_idx < len(chapter_markers) - 1 and int(token[2].text) == chapter_markers[chapter_idx + 1]:
                            chapter_idx += 1
                            chapter_counts.append({})
                            chapter_dict = chapter_counts[chapter_idx]

                        word = token[0].text
                        word_list = [word]
                        if word in ngrams:
                            if word in sentence_dict:
                                sentence_dict[word] += 1
                            else:
                                sentence_dict[word] = 1
                            if word in chapter_dict:
                                chapter_dict[word] += 1
                            else:
                                chapter_dict[word] = 1
                            if word in paragraph_dict:
                                paragraph_dict[word] += 1
                            else:
                                paragraph_dict[word] = 1
                        for i in range(1, len(pwords)+1):
                            word_list.insert(0, pwords[-i])
                            word_tuple = tuple(word_list)
                            if word_tuple in ngrams:
                                if word_tuple in sentence_dict:
                                    sentence_dict[word_tuple] += 1
                                else:
                                    sentence_dict[word_tuple] = 1
                                if word_tuple in chapter_dict:
                                    chapter_dict[word_tuple] += 1
                                else:
                                    chapter_dict[word_tuple] = 1
                                if word_tuple in paragraph_dict:
                                    paragraph_dict[word_tuple] += 1
                                else:
                                    paragraph_dict[word_tuple] = 1
                        pwords.popleft()
                        pwords.append(word)

    # convert sentence, paragraph, chapter count dicts into matrices (sparse)
    v_st = DictVectorizer(sparse=True)
    v_pg = DictVectorizer(sparse=True)
    v_cp = DictVectorizer(sparse=True)
    sentence_mat = v_st.fit_transform(sentence_counts)
    paragraph_mat = v_pg.fit_transform(paragraph_counts)
    chapter_mat = v_cp.fit_transform(chapter_counts)

    # marginalization to get total sentence, paragraph, chapter frequencies
    uform_st_mat = sentence_mat
    uform_pg_mat = paragraph_mat
    uform_cp_mat = chapter_mat
    uform_st_mat[uform_st_mat > 0] = 1.0
    uform_pg_mat[uform_pg_mat > 0] = 1.0
    uform_cp_mat[uform_cp_mat > 0] = 1.0
    norm_st_mat = uform_st_mat.sum(axis=0)
    norm_pg_mat = uform_pg_mat.sum(axis=0)
    norm_cp_mat = uform_cp_mat.sum(axis=0)

    # matrix multiplication to get co-occurrence sentence, paragraph, chapter matrices
    # TODO: might want to store these on disk since computation is expensive
    sentence_cooc = (sentence_mat.T).dot(sentence_mat)
    paragraph_cooc = (paragraph_mat.T).dot(paragraph_mat)
    chapter_cooc = (chapter_mat.T).dot(chapter_mat)

    # unconcern with number of times co-occurring in all sentences, paragraphs, chapters...
    sentence_cooc[sentence_cooc > 0] = 1.0
    paragraph_cooc[paragraph_cooc > 0] = 1.0
    chapter_cooc[chapter_cooc > 0] = 1.0

    # marginalization to get total sentence, paragraph, chapter co-occurrences for each candidate
    normalized_st_cooc = sentence_cooc.sum(axis=1)
    normalized_pg_cooc = paragraph_cooc.sum(axis=1)
    normalized_cp_cooc = chapter_cooc.sum(axis=1)

    # map matrix values back to feature dict
    st_index = {v: k for k, v in v_st.vocabulary_.items()}
    pg_index = {v: k for k, v in v_pg.vocabulary_.items()}
    cp_index = {v: k for k, v in v_cp.vocabulary_.items()}
    for idx in st_index:
        ngram = st_index[idx]
        st_count = norm_st_mat[0, idx]
        st_cooc = normalized_st_cooc[idx, 0]
        features = ngrams[ngram]
        features["count_st"] = st_count
        features["cooc_st"] = st_cooc
    for idx in pg_index:
        ngram = pg_index[idx]
        pg_count = norm_pg_mat[0, idx]
        pg_cooc = normalized_pg_cooc[idx, 0]
        features = ngrams[ngram]
        features["count_pg"] = pg_count
        features["cooc_pg"] = pg_cooc
    for idx in cp_index:
        ngram = cp_index[idx]
        cp_count = norm_cp_mat[0, idx]
        cp_cooc = normalized_cp_cooc[idx, 0]
        features = ngrams[ngram]
        features["count_cp"] = cp_count
        features["cooc_cp"] = cp_cooc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract candidate characters and feature values")
    parser.add_argument('-f', '--file', nargs=1, required=True, help='Book coreNLP file to extract candidates from')
    parser.add_argument('-rf', '--rawfile', nargs=1, required=True, help='Book raw text file to extract candidates from')

    args = vars(parser.parse_args())
    raw_text = args['rawfile'][0]
    markers = section(raw_text)

    tree = ET.parse(args['file'][0])
    candidates = get_candidates(tree, markers)

    ugram_candidates = {cand: v for cand,v in candidates.items() if not isinstance(cand, tuple)}
    bgram_candidates = {cand: v for cand,v in candidates.items() if len(cand) == 2 and isinstance(cand, tuple)}
    tgram_candidates = {cand: v for cand,v in candidates.items() if len(cand) == 3 and isinstance(cand, tuple)}
    sorted_ugrams = sorted(ugram_candidates.items(), key=operator.itemgetter(1))
    sorted_bgrams = sorted(bgram_candidates.items(), key=operator.itemgetter(1))
    sorted_tgrams = sorted(tgram_candidates.items(), key=operator.itemgetter(1))
    for i in range(len(sorted_ugrams)):
        key, count = sorted_ugrams[i][0], sorted_ugrams[i][1]
        print "{0}: {1}".format(key, count)
    for i in range(len(sorted_bgrams)):
        key, count = sorted_bgrams[i][0], sorted_bgrams[i][1]
        print "{0}: {1}".format(key, count)
    for i in range(len(sorted_tgrams)):
        key, count = sorted_tgrams[i][0], sorted_tgrams[i][1]
        print "{0}: {1}".format(key, count)

    print "\n"
    print "".join(["-"] * 10 + ["TAGGING"] + ["-"] * 10)
    get_tag_features(tree, candidates)
    for i in range(len(sorted_ugrams)):
        key = sorted_ugrams[i][0]
        features = candidates[key]
        print "{0}: {1}".format(key, features)
    for i in range(len(sorted_bgrams)):
        key = sorted_bgrams[i][0]
        features = candidates[key]
        print "{0}: {1}".format(key, features)
    for i in range(len(sorted_tgrams)):
        key = sorted_tgrams[i][0]
        features = candidates[key]
        print "{0}: {1}".format(key, features)

    print "\n"
    print "".join(["-"] * 10 + ["CONTAINMENT"] + ["-"] * 10)
    get_containment_features(candidates)
    for i in range(len(sorted_ugrams)):
        key = sorted_ugrams[i][0]
        features = candidates[key]
        print "{0}: {1}".format(key, features)
    for i in range(len(sorted_bgrams)):
        key = sorted_bgrams[i][0]
        features = candidates[key]
        print "{0}: {1}".format(key, features)
    for i in range(len(sorted_tgrams)):
        key = sorted_tgrams[i][0]
        features = candidates[key]
        print "{0}: {1}".format(key, features)

    print "\n"
    print "".join(["-"] * 10 + ["COUNTING"] + ["-"] * 10)
    get_count_features(tree, candidates, markers)
    for i in range(len(sorted_ugrams)):
        key = sorted_ugrams[i][0]
        features = candidates[key]
        print "{0}: {1}".format(key, features)
    for i in range(len(sorted_bgrams)):
        key = sorted_bgrams[i][0]
        features = candidates[key]
        print "{0}: {1}".format(key, features)
    for i in range(len(sorted_tgrams)):
        key = sorted_tgrams[i][0]
        features = candidates[key]
        print "{0}: {1}".format(key, features)

