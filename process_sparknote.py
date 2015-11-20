import os, re, unicodedata
import xml.etree.ElementTree as ET
from django.utils.encoding import smart_str, smart_unicode
from subprocess import call

def normalize(c):
  return unicodedata.normalize('NFD', c.decode('utf-8')).replace(u"\u201c", "\"").replace(u"\u201d", "\"").replace(u"\u2019","\'")

def split_characters(charas):
    charas = charas.split(' and ')
    output = []
    for c in charas:
        output += c.split(',')
    return [c.strip() for c in output if c!='']

def preprocess(book):
    book_dir = books_dir + '/' + book
    chara_files = filter(lambda fname: not (fname.endswith("xml") or fname=="combined" or fname.startswith('.')), os.listdir(book_dir))
    chara_dict = {}
    file_dict = {}
    for file in chara_files:
        with open(book_dir + "/" + file, 'r') as f:
            description = f.readline()
            # might have form A, B and C
            charas = split_characters(normalize(file))
            file_dict[file] = charas
            for chara in charas:
                chara_dict[chara] = (file, chara)

    # find nickname
    nicknames = []
    for chara in chara_dict:
        lparen = chara.find("(")
        if not lparen == -1:
            rparen = chara.find(")")
            nickname = chara[lparen+1:rparen]
            newname = chara[0:lparen] + chara[rparen+1:]
            newname = " ".join(newname.split())
            nicknames.append((chara, newname, nickname))

    for (chara, newname, nickname) in nicknames:
        chara_dict[chara] = (chara_dict[chara][0], newname)
        chara_dict[newname] = chara_dict[chara]
        chara_dict[nickname] = chara_dict[chara]
    return (file_dict, chara_dict)

def run_nlp(book):
    chara_file = open("chara_file.txt", "w")
    all_charas = filter(lambda fname: not (fname.endswith("xml") or fname=="combined" or fname.startswith('.')), os.listdir(books_dir + "/" + book))
    all_charas = [(books_dir + "/" + book + "/" + chara) for chara in all_charas]
    chara_file.write('\n'.join(all_charas))
    chara_file.close()
    call(["bash", nlp_dir + "/corenlp.sh", "--annotators", "tokenize,ssplit,pos,lemma,ner,depparse,parse,dcoref", "-filelist", "chara_file.txt", "-outputDirectory", books_dir + "/" + book])

def get_nicknames(book, files, charas):
    # first pass to add nicknames
    for file in files:
        nlp = ET.parse(books_dir + "/" + book + "/" + file + ".xml")
        corefs = nlp.getroot()[0].find('coreference')
        if corefs == None:
            print '%s no coreferences!' % file
            continue
        for coref in corefs.findall('coreference'):
            chara = ""
            for mention in coref.findall('mention'):
                if mention[4].text in charas:
                    chara = mention[4].text
            if chara != "":
                for mention in coref.findall('mention'):
                    if is_name(mention[4].text) and mention[4].text not in charas:
                        charas[mention[4].text] = charas[chara]
    return charas
            
def is_pronoun(token):
    pronouns = ['he', 'she', 'him', 'her', 'his', 'her', 'herself', 'himself', \
                'He', 'She', 'Him', 'Her', 'His', 'Her', 'Herself', 'Himself' ]
    return token in pronouns

def is_name(s):
    tokens = s.split()
    return sum([is_pronoun(t) for t in tokens]) == 0

def add_relation(relations, c1, c2):
    if c1 != c2:
        cmin = min(c1, c2)
        cmax = max(c1, c2)
        relations.add((cmin, cmax))

def get_relations(book, files, charas):
    relations = set()
    for file in files:
        with open(books_dir + "/" + book + '/' + file, 'r') as f:
            description = normalize(f.readline())
        # if a filename contains several characters, add relations for each two
        if len(files[file]) > 1:
            for c1 in files[file]:
                for c2 in files[file]:
                    add_relation(relations, c1, c2)
        # find mentions in description
        for c2 in charas:
            if description.find(c2) != -1:
                for c1 in files[file]:
                    add_relation(relations, c1, charas[c2][1])
    return relations
                

def process(book):
    (file_dict, chara_dict) = preprocess(book)
    #run_nlp(book)
    chara_dict = get_nicknames(book, file_dict, chara_dict)
    rel_pred = get_relations(book, file_dict, chara_dict)
    rel_true = set()
    with open('annotations/%s.tag'%book) as f:
        line = f.readline()
        while line != "":
            rel = line.split(";")
            assert rel[0] in chara_dict, "%s not a character" % rel[0]
            assert rel[1] in chara_dict, "%s not a character" % rel[0] 
            assert chara_dict[rel[0]][1] == rel[0], "character %s should be %s" % (rel[0], chara_dict[rel[0]][1])
            assert chara_dict[rel[1]][1] == rel[1], "character %s should be %s" % (rel[1], chara_dict[rel[1]][1])
            cmin = min(rel[0], rel[1]) 
            cmax = max(rel[0], rel[1]) 
            rel_true.add((cmin, cmax))
            line = f.readline()

    false_pos = [c for c in rel_pred if c not in rel_true]
    false_neg = [c for c in rel_true if c not in rel_pred]
    print "false positives"
    print false_pos
    print "false negatives"
    print false_neg
    precision = 1-len(false_pos)/float(len(rel_pred))
    recall = 1-len(false_neg)/float(len(rel_true))
    print "Recall %f Precision %f" % (recall, precision)

if __name__ == "__main__":
    books_dir = "books"
    nlp_dir = "../coreNLP"
    process('tess')

