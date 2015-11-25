import os, re, unicodedata, string
import xml.etree.ElementTree as ET
from django.utils.encoding import smart_str, smart_unicode
from subprocess import call
from optparse import OptionParser

def normalize(c):
  return unicodedata.normalize('NFD', c.decode('utf-8')).replace(u"\u201c", "\"").replace(u"\u201d", "\"").replace(u"\u2019","\'").encode('ASCII', 'ignore') #ignore accents

def split_characters(charas):
    charas = charas.split(' and ')
    if len(charas) == 1:
        return charas
    output = []
    for c in charas:
        output += c.split(',')
    return [c.strip() for c in output if c!='']

# get nickname from filenames
def preprocess(book):
    with open(books_dir + '/' + book + '/characters') as f:
        chara_files = [s.strip() for s in f.readlines()]
    names_dict = {}
    file_dict = {}
    chara_dict = {}
    for file in chara_files:
        # might have form A, B and C
        charas = split_characters(normalize(file))
        file_dict[file] = charas
        for chara in charas:
            names_dict[chara] = chara
            chara_dict[chara] = (file, [])

    for chara in names_dict:
        lparen = chara.find("(")
        if not lparen == -1:
            rparen = chara.find(")")
            nickname = chara[lparen+1:rparen]
            newname = chara[0:lparen] + chara[rparen+1:]
            newname = " ".join(newname.split())
            chara_dict[chara][1].append(newname)
            chara_dict[chara][1].append(nickname)
    
    for chara in chara_dict:
        for name in chara_dict[chara][1]:
            names_dict[name] = names_dict[chara]
    return (file_dict, chara_dict, names_dict)

def run_nlp(book):
    chara_file = open("chara_file.txt", "w")
    all_charas = filter(lambda fname: not (fname.endswith("xml") or fname.startswith('.') or fname=='characters'), os.listdir(books_dir + "/" + book))
    all_charas = [(books_dir + "/" + book + "/" + chara) for chara in all_charas]
    chara_file.write('\n'.join(all_charas))
    chara_file.close()
    call(["bash", nlp_dir + "/corenlp.sh", "--annotators", "tokenize,ssplit,pos,lemma,ner,depparse,parse,dcoref", "-filelist", "chara_file.txt", "-outputDirectory", books_dir + "/" + book])

def get_paragraphs(book, files):
    start = 0
    end = 0
    with open(books_dir + "/" + book + "/characters") as f:
        while True:
            character = f.readline().strip()
            if character=='':
                break
            nlp = ET.parse(books_dir + "/" + book + "/" + character + ".xml")
            sentences = nlp.getroot()[0].find('sentences').findall('sentence')
            start = end
            end = start + len(sentences)
            files[character] = (files[character], [start, end])
        nlp = ET.parse(books_dir + "/" + book + "/combined.xml")
        sentences = nlp.getroot()[0].find('sentences').findall('sentence')
        #assert len(sentences)==end, "combined vs separate: %d vs %d" % (len(sentences), end)
    return files

def resolve_reference(name, files, charas):
    candidates = []
    for chara in charas:
        if is_reference(name, chara):
            candidates.append(chara)
    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) == 0:
        print "unresolved references %s no potential references" % (name)
        return ''
    else:
        # choose the one appears the first
        minloc = 100000
        c = ''
        for chara in candidates:
            file = charas[chara][0]
            loc = files[file][1][0]
            if loc < minloc:
                c = chara
                minloc = loc
        print "unresolved references %s resolved to %s among candidates %s" % (name, c, str(candidates))
        return c

def is_subset(shortened, full):
    tokens1 = shortened.split()
    tokens2 = full.split()
    return all(t in tokens2 for t in tokens1)

def is_reference(shortened, full):
    tokens = shortened.split()
    if shortened.startswith('Mr.') or shortened.startswith('Mrs.'):
        return is_subset(' '.join(tokens[1:]), full)
    else:
        return is_subset(shortened, full)

def get_nicknames(book, files, charas, names):
    nlp = ET.parse(books_dir + "/" + book + "/combined.xml")
    corefs = nlp.getroot()[0].find('coreference')
    if corefs==None:
        print "No corefs in %s" % file
        return names
    for coref in corefs.findall('coreference'):
        refs = set()
        unresolved = set()
        for mention in coref.findall('mention'):
            if is_name(mention[4].text):
                name = get_name(mention[4].text)
                if name in names:
                    refs.add(names[name])
                elif is_shortened_name(name):
                    unresolved.add(name)
        if len(refs) == 1:
            chara = refs.pop()
            for mention in coref.findall('mention'):
                if is_name(mention[4].text):
                    name = get_name(mention[4].text)
                    if name not in names:
                        names[name] = names[chara]
                        charas[chara][1].append(name)
                    elif names[name] != names[chara]:
                        if verbose:
                            print "%s could refer to both %s or %s" % (name, names[name], names[chara])
        elif len(unresolved) > 0:
            for name in unresolved:
                chara = resolve_reference(name, files, charas)
                if chara != '':
                    names[name] = names[chara]
                    charas[chara][1].append(name)
        
        if len(refs) > 1 and verbose:
            print "%s refer to the same person" % ", ".join(refs)

    return charas, names
            
def is_pronoun(token):
    pronouns = ['he', 'she', 'him', 'her', 'his', 'her', 'herself', 'himself', 'it', 'its', 'itself', 'they', 'their', 'them', 'themselves',\
                'He', 'She', 'Him', 'Her', 'His', 'Her', 'Herself', 'Himself', 'It', 'Its', 'Itself', 'They', 'Their', 'Them', 'Themselves']
    return token in pronouns

def is_capitalized(token):
    return token[0] in string.ascii_uppercase

def is_determiner(token):
    return token in ['An', 'A', 'The', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten']

def is_name(s):
    tokens = s.split()
    return all([not is_pronoun(t) for t in tokens]) and any(is_capitalized(t) and not is_determiner(t) for t in tokens) and len(tokens) <= 5

def is_shortened_name(s):
    return is_name(s) and len(s.split()) <= 2

def get_name(s):
    if s.endswith(" \'s"):
        s = s[:len(s)-3]
    return s

def add_relation(relations, c1, c2):
    if c1 != c2:
        cmin = min(c1, c2)
        cmax = max(c1, c2)
        relations.add((cmin, cmax))

def get_relations(book, files, names):
    relations = set()
    for file in files:
        with open(books_dir + "/" + book + '/' + file, 'r') as f:
            description = normalize(f.readline())
        # if a filename contains several characters, add relations for each two
        if len(files[file][0]) > 1:
            for c1 in files[file][0]:
                for c2 in files[file][0]:
                    add_relation(relations, c1, c2)
        # find mentions in description
        for c2 in names:
            if description.find(c2) != -1:
                for c1 in files[file][0]:
                    add_relation(relations, c1, names[c2])
    return relations
                

def process(book):
    print "Process book %s" % book
    (file_dict, chara_dict, names_dict) = preprocess(book)
    file_dict = get_paragraphs(book, file_dict)
    (chara_dict, names_dict) = get_nicknames(book, file_dict, chara_dict, names_dict)
    if verbose:
        print chara_dict
    rel_pred = get_relations(book, file_dict, names_dict)
    rel_true = set()
    with open('annotations/%s.tag'%book) as f:
        while True:
            line=f.readline()
            if line== "":
                break
            rel = [normalize(x) for x in line.split(";")]
            assert rel[0] in names_dict, "%s not a character" % rel[0]
            assert rel[1] in names_dict, "%s not a character" % rel[1] 
            assert names_dict[rel[0]] == rel[0], "character %s should be %s" % (rel[0], names_dict[rel[0]])
            assert names_dict[rel[1]] == rel[1], "character %s should be %s" % (rel[1], names_dict[rel[1]])
            cmin = min(rel[0], rel[1]) 
            cmax = max(rel[0], rel[1]) 
            rel_true.add((cmin, cmax))

    false_pos = [c for c in rel_pred if c not in rel_true]
    false_neg = [c for c in rel_true if c not in rel_pred]
    if verbose:
        print "false positives"
        print false_pos
        print "false negatives"
        print false_neg
    precision = 1-len(false_pos)/float(len(rel_pred))
    recall = 1-len(false_neg)/float(len(rel_true))
    print "Recall %f Precision %f" % (recall, precision)
    return (recall, precision)

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-b", "--book", dest="book", help="which book to process", default="all")
    parser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False)
    parser.add_option("-n", "--run_nlp", dest="run_nlp", action="store_true", default=False)

    (options, args) = parser.parse_args()
    books_dir = "books"
    nlp_dir = "../coreNLP"
    verbose = options.verbose
    if options.book == 'all':
        recall = 0
        precision = 0
        annotated = filter(lambda fname: fname.endswith("tag"), os.listdir("annotations"))
        annotated = [name[:len(name)-4] for name in annotated]
        for book in annotated:
            if options.run_nlp:
                run_nlp(book)
            (r, p) = process(book)
            recall += r
            precision += p
        print "Average Recall %f Precision %f" % (recall/len(annotated), precision/len(annotated))
    else:
        if options.run_nlp:
            run_nlp(options.book)
        process(options.book)

