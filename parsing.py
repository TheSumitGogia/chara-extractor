import os
from subprocess import call

book_dir = "./books"
nlp_dir = "/home/summit/Downloads/stanford-corenlp-full-2015-04-20"
all_books = os.listdir(book_dir)
for dirpath in all_books:
    all_charas = os.listdir(book_dir + "/" + dirpath)
    all_charas = [(book_dir + "/" + dirpath + "/" + chara) for chara in all_charas]
    chara_file = open("chara_file.txt", "w")
    chara_file.write("\n".join(all_charas))
    chara_file.close()
    call(["bash", nlp_dir + "/corenlp.sh", "--annotators", "tokenize,ssplit,pos,lemma,ner,depparse,parse,dcoref", "-filelist", "chara_file.txt", "-outputDirectory", book_dir + "/" + dirpath])
