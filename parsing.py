import os
import argparse
from subprocess import call

def run_spark_nlp(nlp_dir, book_dir, out_dir):
    all_books = os.listdir(book_dir)
    for dirpath in all_books:
        all_charas = os.listdir(book_dir + "/" + dirpath)
        all_charas = [(book_dir + "/" + dirpath + "/" + chara) for chara in all_charas]
        all_charas = filter(lambda fname: not (fname.endswith("xml")), all_charas)
        chara_file = open("chara_file.txt", "w")
        chara_file.write("\n".join(all_charas))
        chara_file.close()
        call(["bash", nlp_dir + "/corenlp.sh", "--annotators", "tokenize,ssplit,pos,lemma,ner,depparse,parse,dcoref", "-filelist", "chara_file.txt", "-outputDirectory", out_dir + "/" + dirpath])

def run_book_nlp(nlp_dir, book_dir, out_dir):
    # get all books in book full text dir
    all_books = os.listdir(book_dir)
    all_books = [book_dir + '/' + book for book in all_books]

    # create book name index for batch CoreNLP processing
    book_list = open(book_dir + '/book_list.txt', 'w')
    book_list.write('\n'.join(all_books))
    book_list.close()

    call(["bash", nlp_dir + "/corenlp.sh", "--annotators", "tokenize,ssplit,pos,lemma,ner", "-filelist", book_dir + "/book_list.txt", "-outputDirectory", out_dir])

    os.remove(book_dir + '/book_list.txt')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Stanford NLP tools on Sparknotes data")
    parser.add_argument('-t', '--type', nargs=1, type=int, required=True, help='Annotation filetype')
    parser.add_argument('-d', '--nlpdir', nargs=1, required=True, help='Stanford NLP root directory')
    parser.add_argument('-b', '--booksdir', nargs=1, required=True, help='Book character lists directory')
    parser.add_argument('-o', '--outdir', nargs=1, required=True, help='Annotation output directory')

    args = vars(parser.parse_args())
    filetype = args['type'][0]
    book_dir = args['booksdir'][0]
    out_dir = args['outdir'][0]
    nlp_dir = args['nlpdir'][0]

    if filetype == 0:
        run_spark_nlp(nlp_dir, book_dir, out_dir)
    elif filetype == 1:
        run_book_nlp(nlp_dir, book_dir, out_dir)
