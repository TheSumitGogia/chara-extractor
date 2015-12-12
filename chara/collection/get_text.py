from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers
from unidecode import unidecode
import os
import string

#OUT_PATH = 'raw_texts/%s.txt'

def process_file(filename, outdir):
    outpath = outdir + '/%s.txt'
    with open(filename) as f:
        for line in f:
          spl = line.split('|')
          book = spl[0]
          uids = map(int, spl[3].strip(string.lowercase + '\n').split(','))
          try:
            with open(outpath % book, 'w') as out:
              for uid in uids:
                raw_text = load_etext(uid)
                try:
                  text = strip_headers(unidecode(raw_text.encode('latin-1').decode('utf-8')))
                except UnicodeDecodeError:
                  text = strip_headers(raw_text)
                out.write(text.encode('utf-8'))
          except ValueError as e:
            print '%s|%s' % (book, uid), e
            os.remove(outpath % book)

def download(
        goodfile='good.txt',
        uglyfixfile=None,
        outdir='raw'):

    process_file(goodfile, outdir)
    if uglyfixfile is not None:
        process_file(uglyfixfile, outdir)
