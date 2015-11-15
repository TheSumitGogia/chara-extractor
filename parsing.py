import os
import argparse
from subprocess import call

def run_nlp(nlp_dir, book_dir):
    all_books = os.listdir(book_dir)
    for dirpath in all_books:
        all_charas = os.listdir(book_dir + "/" + dirpath)
        all_charas = [(book_dir + "/" + dirpath + "/" + chara) for chara in all_charas]
        all_charas = filter(lambda fname: not (fname.endswith("xml")), all_charas)
        chara_file = open("chara_file.txt", "w")
        chara_file.write("\n".join(all_charas))
        chara_file.close()
        call(["bash", nlp_dir + "/corenlp.sh", "--annotators", "tokenize,ssplit,pos,lemma,ner,depparse,parse,dcoref", "-filelist", "chara_file.txt", "-outputDirectory", book_dir + "/" + dirpath])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Stanford NLP tools on Sparknotes data")
    parser.add_argument('-d', '--nlpdir', nargs=1, required=True, help='Stanford NLP root directory')
    parser.add_argument('-b', '--booksdir', nargs=1, required=True, help='Book character lists directory')

    args = vars(parser.parse_args())
    book_dir = args['booksdir'][0]
    nlp_dir = args['nlpdir'][0]

    run_nlp(nlp_dir, book_dir)
