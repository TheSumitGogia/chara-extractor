import xml.etree.ElementTree as ET
import operator
import argparse
import itertools as it
from collections import deque

def get_candidates(tree):
    cutoff_u = 30
    cutoff_b = 30
    cutoff_t = 10
    ugrams = {}
    bgrams = {}
    tgrams = {}

    root = tree.getroot()
    for document in root:
        for sentences in document:
            for sentence in sentences:
                for tokens in sentence:
                    pword = ""
                    ppword = ""
                    for token in tokens:
                        noun = (token[4].text.startswith('NN'))

                        if noun:
                            word = token[0].text
                            if word in ugrams:
                                ugrams[word] += 1
                            else:
                                ugrams[word] = 1
                            if not pword == '':
                                if (pword, word) in bgrams:
                                    bgrams[(pword, word)] += 1
                                else:
                                    bgrams[(pword, word)] = 1
                            if not ppword == '':
                                if (ppword, pword, word) in tgrams:
                                    tgrams[(ppword, pword, word)] = 1
                                else:
                                    tgrams[(ppword, pword, word)] = 1
                            ppword = pword
                            pword = word

    ugrams = { ugram: ugrams[ugram] for ugram in ugrams.keys() if ugram[0].isupper() }
    bgrams = { bgram: bgrams[bgram] for bgram in bgrams.keys() if bgram[1][0].isupper() }
    tgrams = { tgram: tgrams[tgram] for tgram in tgrams.keys() if tgram[2][0].isupper() }
    sorted_ugrams = sorted(ugrams.items(), key=operator.itemgetter(1))
    sorted_bgrams = sorted(bgrams.items(), key=operator.itemgetter(1))
    sorted_tgrams = sorted(tgrams.items(), key=operator.itemgetter(1))

    filtered_ugrams = sorted_ugrams[-cutoff_u:]
    filtered_bgrams = sorted_bgrams[-cutoff_b:]
    filtered_tgrams = sorted_tgrams[-cutoff_t:]

    return (filtered_ugrams, filtered_bgrams, filtered_tgrams)

def get_tag_features(tree, ngrams):
    # features
    # 1) average last ner percentage
    # 2) average all ner percentage
    # 3) average capitalization percentage

    # get max gram length for tracking
    max_gram = max(map(lambda x: len(x), ngrams.keys()))

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
                        ner = 1 if token[0].text == "MISC" or token[0].text == "PERSON" else 0
                        ner_list = [ner]
                        caps = 1 if token[0].text[0].isupper() else 0
                        caps_list = [caps]
                        for i in range(1, len(pwords)+1):
                            word_list.insert(0, pwords[-i])
                            ner_list.insert(0, pner[-i])
                            caps_list.insert(0, pcaps[-i])
                            word_tuple = tuple(word_list)
                            if word_tuple in ngrams:
                                features = ngrams[word_tuple]
                                features['avg_ner'] += ((sum(ner_list) * 1.0 / len(ner_list)) / features['count'])
                                features['avg_last_ner'] += (ner / features['count'])
                                features['avg_cap'] += ((sum(caps_list) * 1.0 / len(caps_list)) / features['count'])
                                features['avg_last_cap'] += (caps / features['count'])
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
        if len(ngram) > 1:
            cont_features = ngrams[ngram]
            for r in range(1, len(ngram)):
                subs = it.combinations(ngram, r)
                for sub in subs:
                    if sub in ngrams:
                        features = ngrams[sub]
                        features["contained"] = 1
                        features["container_count"] += cont_features["count"]
                        if cont_features["contains"] == 0:
                            cont_features["contains"] = 1
                        cont_features["contains_count"] += features["count"]

def get_count_features(tree, ngrams):
    sentence_idx = -1
    sentence_counts = []
    chapter_idx = -1
    chapter_counts = []
    paragraph_idx = -1
    paragraph_counts = []

    sentence_dict = {}
    paragraph_dict = {}
    chapter_dict = {}

    root = tree.getroot()
    for document in root:
        for sentences in document:
            for sentence in sentences:
                sentence_idx += 1
                sentence_counts.append(sentence_idx)
                sentence_dict = sentence_counts[sentence_idx]
                for tokens in sentence:
                    pwords = deque([""] * (max_gram - 1))
                    for token in tokens:
                        if token[2] == paragraph_markers[paragraph_idx + 1]:
                            paragraph_idx += 1
                            paragraph_counts.append({})
                            paragraph_dict = paragraph_counts[paragraph_idx]
                        if token[2] == chapter_markers[chapter_idx + 1]:
                            chapter_idx += 1
                            chapter_counts.append({})
                            chapter_dict = chapter_counts[chapter_idx]

                        word = token[0].text
                        word_list = [word]
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
    norm_st_mat = sum(uform_st_mat, axis=0)
    norm_pg_mat = sum(uform_pg_mat, axis=0)
    norm_cp_mat = sum(uform_cp_mat, axis=0)

    # matrix multiplication to get co-occurrence sentence, paragraph, chapter matrices
    # TODO: might want to store these on disk since computation is expensive
    sentence_cooc = (sentence_mat.T).dot(sentence_mat)
    paragraph_cooc = (paragraph_cooc.T).dot(paragraph_mat)
    chapter_cooc = (chapter_cooc.T).dot(chapter_mat)

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
        st_count = normalized_st_mat[0, idx]
        st_cooc = normalized_st_cooc[idx, 0]
        features = ngrams[ngram]
        features["count_st"] = st_count
        features["cooc_st"] = st_cooc
    for idx in pg_index:
        ngram = pg_index[idx]
        pg_count = normalized_pg_mat[0, idx]
        pg_cooc = normalized_pg_cooc[idx, 0]
        features = ngrams[ngram]
        features["count_pg"] = pg_count
        features["cooc_pg"] = pg_cooc
    for idx in cp_index:
        ngram = cp_index[idx]
        cp_count = normalized_cp_mat[0, idx]
        cp_cooc = normalized_cp_cooc[idx, 0]
        features = ngrams[ngram]
        features["count_cp"] = cp_count
        features["cooc_cp"] = cp_cooc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract candidate characters and feature values")
    parser.add_argument('-f', '--file', nargs=1, required=True, help='Book text file to extract candidates from')

    args = vars(parser.parse_args())
    tree = ET.parse(args['file'][0])
    candidates = get_candidates(tree)

    print "--Filtered Unigrams"
    for i in range(len(candidates[0])):
        ugram = candidates[0][i]
        print "{0}: {1}".format(ugram[0], ugram[1])
    print "--Filtered Bigrams"
    for i in range(len(candidates[1])):
        bgram = candidates[1][i]
        print "{0}: {1}".format(bgram[0], bgram[1])
    print "--Filtered Trigrams"
    for i in range(len(candidates[2])):
        tgram = candidates[2][i]
        print "{0}: {1}".format(tgram[0], tgram[1])

