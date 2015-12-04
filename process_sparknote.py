import os, re, unicodedata, string
import xml.etree.ElementTree as ET
from django.utils.encoding import smart_str, smart_unicode
from subprocess import call
from optparse import OptionParser
from disambiguation import *

class Paragraph:
    def __init__(self, characters, index):
        self.index = index
        self.start = -1 #start sentence
        self.end = -1 #end sentence
        self.characters = characters
        self.mentions = {}

class Character:
    def __init__(self, sname, pi):
        self.sname = sname
        self.paragraph = pi
        self.references = []
        self.sex = ''

        lparen = sname.find("(")
        if not lparen == -1:
            rparen = sname.find(")")
            nickname = sname[lparen+1:rparen]
            name = sname[0:lparen].split() + sname[rparen+1:].split()

            name1 = " ".join(name)
            # nickname + lastname
            name[0] = nickname
            name2 = " ".join(name)

            self.references.append(nickname)
            self.references.append(name1)
            self.references.append(name2)

    def set_sex(self, sex):
        if self.sex != '':
            assert self.sex == sex, 'Condtradiction in %s\'s sex' % self.sname
        self.sex = sex

class Book:
    def __init__(self, name):
        self.name = name
        self.paragraphs = []
        self.characters = {}
        self.references = {}
        self.max_character_num_tokens = 0

    def __init__(self, name, paragraphs, characters, references):
        self.name = name
        self.paragraphs = paragraphs
        self.characters = characters
        self.references = references
        self.max_character_num_tokens = 0
        for character in characters:
            l = len(re.split('(\W+)', character))
            if l > self.max_character_num_tokens:
                self.max_character_num_tokens = l

    def add_reference(self, character, reference):
        self.references[reference] = character
        self.characters[character].references.append(reference)

    def add_mention(self, character, sentence):
        for p in self.paragraphs:
            s = int(sentence.get('id'))-1
            if p.start <= s and p.end > s:
                if character not in p.characters:
                    if s not in p.mentions:
                        p.mentions[s] = [character]
                    else:
                        p.mentions[s].append(character)
                return


def normalize(c): 
    return unicodedata.normalize('NFD', c.decode('utf-8')).replace(u"\u201c", "\"").replace(u"\u201d", "\"").replace(u"\u2019","\'").replace(u"\u2014", "-").encode('ASCII', 'ignore') #ignore accents

def split_characters(charas):
    if charas.find(' and ') != -1:
        characters = charas.replace(' and ', ',').split(',')
        characters = [c.strip() for c in characters if c!='']
        for i in range(len(characters)):
            if characters[i] in ALL_TITLES:
                lastname = characters[i+1].split()[-1]
                characters[i] += " " + lastname
        return characters
    else:
        return [charas]

def get_character_files(book):
    with open(books_dir + '/' + book + '/characters') as f:
        chara_files = [s.strip() for s in f.readlines()]
    return chara_files

# deal with 'A, B and C' and 'A(B)'
def preprocess(bookname):
    chara_files = get_character_files(bookname)
    names_dict = {}
    paragraphs = []
    chara_dict = {}
    pi=0
    for file in chara_files: 
        # might have form A, B and C
        charas = split_characters(normalize(file))
        paragraphs.append(Paragraph(charas, pi))
        for chara in charas:
            chara_dict[chara] = Character(chara, pi)
            for name in chara_dict[chara].references:
                names_dict[name] = chara
            names_dict[chara] = chara
        pi+=1
    return Book(bookname, paragraphs, chara_dict, names_dict) 

def run_nlp(books): 
    with open("chara_file.txt", "w") as f:
        all_files = []
        for book in books:
            #all_files = get_character_files(bookname)
            all_files.append(books_dir + "/combined/" + book)
        f.write('\n'.join(all_files))
    call(["bash", nlp_dir + "/corenlp.sh", "--annotators", "tokenize,ssplit,pos,lemma,ner,depparse,parse,dcoref", "-filelist", "chara_file.txt", "-outputDirectory", books_dir + "/combined"])

# get coresponding paragraphs for each separate files in the combined file
def get_paragraphs_split(book): 
    nlp = ET.parse(books_dir + "/combined/" + book.name + ".xml")
    sentences = nlp.getroot()[0].find('sentences').findall('sentence')
    
    with open(books_dir+'/combined/' + book.name) as f:
        lines = f.readlines()
    lines = map(normalize, lines)

    #normalize may cause slight misalignment because of encoding and decoding
    text = "".join(lines).strip()
    newline_loc = [i.start() for i in re.finditer('\n', text)]
    newline_loc.append(len(text))
    assert len(newline_loc) == len(book.paragraphs), "%d vs %s" % (len(newline_loc), len(book.paragraphs)) 
  
    pi = 0
    si = 0
    offset=0
    for s in sentences:
        tokens = s.find('tokens').findall('token')
        token_text = [text[int(t.find('CharacterOffsetBegin').text)+offset:int(t.find('CharacterOffsetEnd').text)+offset] for t in tokens]
        #print [s.get('id'), token_text]
        end = int(tokens[-1].find('CharacterOffsetEnd').text)
        #print [end+offset, newline_loc[pi]]
        if end+offset - newline_loc[pi] > -4:
            if abs(end+offset-newline_loc[pi]) < 4:
                offset = newline_loc[pi]-end
            book.paragraphs[pi].start = si
            si = int(s.get('id'))
            book.paragraphs[pi].end = si
            #print [pi, [book.paragraphs[pi].start, book.paragraphs[pi].end]]  
            pi+=1

    assert pi == len(book.paragraphs), "%d vs %s" % (pi, len(book.paragraphs))
    if verbose:
        print 'Paragraph Split'
        for p in book.paragraphs:
            print [p.start, p.end]

def find_coref_sex(coref):
    for mention in coref.find('mention'):
        if mention.find('text') in ['he', 'him', 'his', 'himself', 'He', 'Him', 'His', 'Himself']:
            return  'MALE'
        elif mention.find('text') in ['she', 'her', 'herself', 'She', 'Her', 'Herself']:
            return 'FEMALE'
    return ''

def get_mention_text(nlp_tree, mention):
    sentences = nlp_tree.getroot()[0].find('sentences').findall('sentence')
    sentence = sentences[int(mention.find('sentence').text)-1]
    start = int(mention.find('start').text)-1
    end = int(mention.find('end').text)-1
    return get_sentence_text(sentence, start, end)

def get_sentence_text(sentence, start, end):
    tokens = sentence.find('tokens').findall('token') 
    text = ''
    last_end = 0
    for i in range(start, end):
        token = tokens[i]
        c_start = int(token.find('CharacterOffsetBegin').text)
        if c_start > last_end and i != start:
            text += ' '
        text += normalize(token.find('word').text.encode('utf-8'))
        last_end = int(token.find('CharacterOffsetEnd').text)
    return text

def is_valid_mention(nlp_tree, mention, max_num_tokens):
    ts = get_mention_text(nlp_tree, mention).split()
    if len(ts) > max_num_tokens:
        return False
    if len(ts) == 1 and all([c in string.ascii_lowercase for c in ts[0]]):
        return False
    start = int(mention.find('start').text)-1
    sentences = nlp_tree.getroot()[0].find('sentences').findall('sentence')
    sentence = sentences[int(mention.find('sentence').text)-1]
    tokens = sentence.find('tokens').findall('token') 
    # check the token before is not determiner
    if start > 0:
        token_before = tokens[start-1]
        if token_before.find('POS').text == 'DT':
            return False
    # check the first token is not PRP
    return tokens[start].find('POS').text not in ['PRP', 'PRP$']

def resolve_references(book): 
    nlp = ET.parse(books_dir + "/combined/" + book.name + ".xml")
    corefs = nlp.getroot()[0].find('coreference').findall('coreference')
    
    candidates = [tuple(ref.split()) for ref in book.references]
    for coref in corefs: 
        for mention in coref.findall('mention'):
            if is_valid_mention(nlp, mention, book.max_character_num_tokens):
                name = get_mention_text(nlp, mention)
                character = None
                sex = find_coref_sex(coref)
                if name.endswith('\'s'):
                    name = name[:-2].strip()
                if name.endswith('s\''):
                    name = name[:-1].strip()
                if name in book.references:
                    #book.add_mention(book.references[name])
                    character = book.references[name]
                    if sex != '':
                        book.characters[character].set_sex(sex)

                else: # try to resolve 
                    cand = tuple(name.split())
                    if cand[0] in ALL_TITLES:
                        sex = 'MALE' if cand[0] in MALE_TITLES else 'FEMALE'
                    potential_charas = find_potential_references(candidates, cand)
                    potential_charas = set(map(lambda x: book.references[" ".join(x)], potential_charas))

                    # filter out opposite sex
                    if sex != '':
                        if cand[0] in MALE_TITLES:
                            potential_charas = filter(lambda x: book.characters[x].sex != 'FEMALE', potential_charas)
                        if cand[0] in FEMALE_TITLES:
                            potential_charas = filter(lambda x: book.characters[x].sex != 'MALE', potential_charas)

                    if len(potential_charas) == 1:
                        book.add_reference(potential_charas.pop(), name)
                        #book.add_mention(mention, book.references[name])
                        character = book.references[name]
                        if sex != '':
                            book.characters[character].set_sex(sex)
                    elif len(potential_charas) > 1:
                        # resolve to the first one
                        chara = ''
                        for p in book.paragraphs:
                            for c in p.characters:
                                if c in potential_charas:
                                    chara = c
                                    break
                            if chara != '':
                                break
                        book.add_reference(chara, name)
                        #book.add_mention(mention, book.references[name])
                        character = book.references[name]
                        if sex != '':
                            book.characters[character].set_sex(sex)

                        if verbose:
                            print "resolve %s to %s among %s" %(name, character, potential_charas)
                    else:
                        if verbose:
                            print "can't resolve %s" %(name)

# resolve stuff like the Pevensies, the Pevensie children/house/family
def family_resolution(families, text):
    cand = text.split()
    if cand[0] in ['the', 'The']:
        if len(cand) == 3 and cand[1] in families and cand[2] in ['children', 'house', 'family', 'people']:
            return families[cand[1]]
        if len(cand) == 2 and cand[1].endswith('s') and cand[1][:-1] in families:
            return families[cand[1][:-1]]
    return []


# find mentions in a sentence
# handles cases where a sentence Midred Montag will add mention Midred Montag instead of Montag
def find_mention(book, sentence, families):
    tokens = sentence.find('tokens').findall('token')
    min_end = 1
    for start in range(len(tokens)):
        if min_end <= start:
            min_end = start+1
        if min_end > len(tokens):
            break
        mentions = []
        for end in range(min_end, min(len(tokens), start + book.max_character_num_tokens)):
            text = get_sentence_text(sentence, start, end)
            if text in book.references:
                min_end = max(end+1, min_end)
                mentions = [book.references[text]]
            else:
                family = family_resolution(families, text)
                if len(family) > 0:
                    print "resolve family mention %s %s" % (text, family)
                    min_end = max(end+1, min_end)
                    mentions = family
        # only add the last found one, ie, the longest one
        if len(mentions) > 0:
            for mention in mentions:
                book.add_mention(book.references[mention], sentence)

def find_mentions(book):
    nlp = ET.parse(books_dir + "/combined/" + book.name + ".xml")
    sentences = nlp.getroot()[0].find('sentences').findall('sentence')
    last_names = [c.split()[-1] for c in book.characters]
    families = dict([(last_name,[]) for last_name in last_names if last_name[0] in string.ascii_uppercase])
    for c in book.characters:
        if c.split()[-1] in families:
            families[c.split()[-1]].append(c)
        
    for sentence in sentences:
        #print get_sentence_text(sentence, 0, len(sentence.find('tokens').findall('token')))
        find_mention(book, sentence, families)

def add_relation(relations, c1, c2):
    if c1 != c2:
        cmin = min(c1, c2)
        cmax = max(c1, c2)
        relations.add((cmin, cmax))

def get_relations(book):
    if verbose:
        print 'Mentions'
        for p in book.paragraphs:
            print "%s, %s" % (p.characters, p.mentions)
    relations = set()
    for p in book.paragraphs:
        if len(p.characters) > 1:
            for c1 in p.characters:
                for c2 in p.characters:
                    add_relation(relations, c1, c2)
        for s in p.mentions:
            mentions = p.mentions[s]
            for mention in mentions:
                for c in p.characters:
                    add_relation(relations, mention, c)
    return relations

def process(book):
    print "Process book %s" % book
    book = preprocess(book)
    get_paragraphs_split(book)
    resolve_references(book)
    find_mentions(book)
    if verbose:
        print book.references
        print "\n".join(["%s: %s" % (c, ", ".join(book.characters[c].references)) for c in book.characters])
    rel_pred = get_relations(book)
    rel_true = set()

    if writeToFile:
        with open('sparknotes/%s_characters.txt' % book.name, 'w') as f:
            characters = dict([(c, book.characters[c].references) for c in book.characters])
            f.write(str(characters))
        with open('sparknotes/%s_relations.txt' % book.name, 'w') as f:
            f.write(str(rel_pred))
    
    if compare:
        with open('annotations/%s.tag'%book.name) as f:
            while True:
                line=f.readline()
                if line== "":
                    break
                rel = [normalize(x) for x in line.split(";")]
                assert rel[0] in book.characters, "%s not a character" % rel[0]
                assert rel[1] in book.characters, "%s not a character" % rel[1] 
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
    else:
        return (1,1)

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-b", "--book", dest="book", help="which book to process", default="annotated")
    parser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False)
    parser.add_option("-n", "--run_nlp", dest="run_nlp", action="store_true", default=False)
    parser.add_option("-c", "--compare", dest="compare", help='compare with annotations', action="store_true", default=False)

    (options, args) = parser.parse_args()
    books_dir = "books"
    nlp_dir = "../coreNLP"
    verbose = options.verbose
    compare = options.compare
    writeToFile = False
    if options.book == 'annotated':
        recall = 0
        precision = 0
        annotated = filter(lambda fname: fname.endswith("tag"), os.listdir("annotations"))
        annotated = [name[:len(name)-4] for name in annotated]
        unsatisfied = []
        if options.run_nlp:
            run_nlp(annotated)
        for book in annotated:
            (r, p) = process(book)
            if r < 0.7 or p < 0.7:
                unsatisfied.append(book)
            recall += r
            precision += p
        print "Average Recall %f Precision %f" % (recall/len(annotated), precision/len(annotated))
        print unsatisfied
    elif options.book == 'raw':
        writeToFile = True
        raw = filter(lambda fname: fname.endswith("txt"), os.listdir("raw_texts"))
        raw = [name[:len(name)-4] for name in raw]
        if options.run_nlp:
            run_nlp(raw)
        for book in raw:
            process(book)
    else:
        if options.run_nlp:
            run_nlp([options.book])
        process(options.book)
