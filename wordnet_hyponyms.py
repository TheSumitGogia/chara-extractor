from nltk.corpus import wordnet as wn

def get_word(syn):
  return str(syn.name()).split('.')[0].replace('_', ' ')

def enumerate_hyponyms(syn):
  ret = set([])
  for s in syn.hyponyms():
    ret.add(get_word(s))
    ret.update([hyp for hyp in enumerate_hyponyms(s)])
  return ret

root = wn.synsets('person')[0]
hyponyms = enumerate_hyponyms(root)
