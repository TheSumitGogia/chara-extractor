import xml.etree.ElementTree as ET
import operator
import argparse

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

