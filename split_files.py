from zipfile import ZipFile
import os
import sys

RAW_NLP_DIR = 'raw_nlp/'
RAW_TEXT_DIR = 'raw_texts/'
TOKENS_DIR = 'tokens/'

num_partitions = int(sys.argv[1])
partition_ids = range(num_partitions) if len(sys.argv) < 3 else [int(sys.argv[2])]

# Check directory states consistent
nlp_filenames = set(map(lambda f: f.split('.')[0], os.listdir(RAW_NLP_DIR)))
text_filenames = set(map(lambda f: f.split('.')[0], os.listdir(RAW_TEXT_DIR)))
tokens_filenames = set(map(lambda f: f.split('.')[0], os.listdir(TOKENS_DIR)))
assert nlp_filenames == text_filenames == tokens_filenames, '%s and %s directories inconsistent' % (RAW_NLP_DIR, RAW_TEXT_DIR)

num_texts = len(os.listdir(RAW_TEXT_DIR))
for n in partition_ids:
  partition = range(num_texts / num_partitions * n, num_texts / num_partitions * (n + 1))
  nlp_partition = map(sorted(os.listdir(RAW_NLP_DIR)).__getitem__, partition)
  text_partition = map(sorted(os.listdir(RAW_TEXT_DIR)).__getitem__, partition)
  tokens_partition = map(sorted(os.listdir(TOKENS_DIR)).__getitem__, partition)

  with ZipFile('partition%d.zip' % n, 'w') as zf:
    map(lambda f: zf.write(RAW_NLP_DIR + f), nlp_partition)
    map(lambda f: zf.write(RAW_TEXT_DIR + f), text_partition)
    map(lambda f: zf.write(TOKENS_DIR + f), tokens_partition)
