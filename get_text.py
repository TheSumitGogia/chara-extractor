from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers
import string

OUT_PATH = 'raw_texts/%s.txt'

with open('gutenberg/good.txt') as f:
  for line in f:
    spl = line.split('|')
    book = spl[0]
    uid = int(spl[3].strip(string.lowercase + '\n'))
    try:
      text = strip_headers(load_etext(uid))
      with open(OUT_PATH % book, 'w') as out:
        out.write(text.encode('utf-8'))
    except ValueError as e:
      print '%s|%s' % (book, uid), e
