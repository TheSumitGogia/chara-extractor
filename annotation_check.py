# Usage:
# python annotation_check.py book_name
#
# Example:
# python annotation_check.py returnking

import os, sys, unicodedata

book = sys.argv[1]
annotations_path = 'annotations/%s.tag' % book
book_path = 'books/%s' % book

# normalize unicode
def normalize(c):
  return unicodedata.normalize('NFD', c.decode('utf-8'))

characters = set(map(normalize, os.listdir(book_path)))

# character1;character2;
def check_format(ln, s):
  spl = map(normalize, s.split(';'))
  assert len(spl) == 3, 'Invalid format, line %d' % ln
  (c1, c2, c3) = spl
  assert c1 in characters, 'Cannot find character: "%s", line %d' % (c1, ln)
  assert c2 in characters, 'Cannot find character: "%s", line %d' % (c2, ln)

err_count = 0;
with open(annotations_path) as f:
  for (ln, annotation) in enumerate(f):
    try:
      check_format(ln + 1, annotation)
    except AssertionError as e:
      print e
      err_count += 1;
      continue;
print 'Finished checking file %s with %d errors' % (book, err_count)
