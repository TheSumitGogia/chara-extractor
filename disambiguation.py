from collections import deque
from fuzzywuzzy import fuzz, process

def populate_gender_dict():
  d = {}
  with open('stanford/namegender.combine.txt') as f:
    for line in f:
      name, gender = line.rsplit('\t', 1)
      d[name] = gender.strip()
  with open('stanford/male.unigrams.txt') as f:
    for name in f:
      d[name] = 'MALE'
  with open('stanford/female.unigrams.txt') as f:
    for name in f:
      d[name] = 'FEMALE'
  return d

gender_dict = populate_gender_dict()
MALE_TITLES = ['Mr.', 'Mister']
FEMALE_TITLES = ['Mrs.', 'Ms.', 'Miss']
ALL_TITLES = MALE_TITLES + FEMALE_TITLES

def disambiguate(candidates):
  all_maps = {}
  best_maps = {}

  for cand in candidates:
    all_maps[cand] = find_potential_references(candidates, cand)

  # DFS for complete paths
  connected_cands = {cand: dfs(all_maps, cand) for cand in candidates}
  for cand in candidates:
    all_maps[cand].update(connected_cands[cand])

  return all_maps

def find_potential_references(candidates, cand):
  refs = partial_reference(candidates, cand) + \
           nickname_resolution(candidates, cand) + \
           title_resolution(candidates, cand)
  return set(refs)

def contains_tuple(t_outer, t_inner):
  inner_idx=0
  for t in t_outer:
      if t.lower() == t_inner[inner_idx].lower():
          inner_idx+=1
      if inner_idx == len(t_inner):
          return True
  return False

def resolve_title(ocand, cand):
  if cand != ocand and cand[0] in ALL_TITLES and cand[-1] == ocand[-1]:
    first_name = ocand[0].lower()
    if first_name in gender_dict:
      if gender_dict[first_name] == 'MALE' and cand[0] in MALE_TITLES:
        return True
      elif gender_dict[first_name] == 'FEMALE' and cand[0] in FEMALE_TITLES:
        return True
  return False

def fuzzy_match(s1, s2):
  # ignore titles
  return (s1 not in ALL_TITLES and s2 not in ALL_TITLES) and \
         (s1 in s2 or max(fuzz.ratio(s1, s2[:i]) for i in range(len(s2))) >= 70)

def fuzzy_contains_tuple(t_outer, t_inner):
  for i in range(len(t_outer) - len(t_inner) + 1):
    if all(fuzzy_match(a, b) for a, b in zip(t_inner, t_outer[i:i + len(t_inner)])):
      return True
  return False

def score(t):
  title_score = (t[0] in MALE_TITLES) - (t[0] in FEMALE_TITLES)
  first_name = t[0].lower()
  name_score = (first_name in gender_dict and gender_dict[first_name] == 'MALE') - \
                (first_name in gender_dict and gender_dict[first_name] == 'FEMALE')
  return title_score + name_score

def gender_match(t1, t2):
  return abs(score(t1) - score(t2)) < 2

def str_gender_match(s1, s2):
  return gender_match(tuple(s1.split()), tuple(s2.split()))

# (A,) -> (A, B)
def partial_reference(candidates, cand):
  ret = []
  for ocand in candidates:
    if cand != ocand and contains_tuple(ocand, cand):
      ret.append(ocand)
  return ret

# substring
def nickname_resolution(candidates, cand):
  ret = []
  for ocand in candidates:
    if cand != ocand and fuzzy_contains_tuple(ocand, cand):
      ret.append(ocand)
  return ret

# Mr. -> name or Mrs. -> name
def title_resolution(candidates, cand):
  ret = []
  for ocand in candidates:
      if cand != ocand and resolve_title(ocand, cand):
          ret.append(ocand)
  return ret

'''
def fuzzy_wuzzy_resolution(candidates, cand):
  ret = []
  str_candidates = set(' '.join(t) for t in candidates)
  str_cand = ' '.join(cand)
  ocands = filter(lambda t: t[1] >= 70 and str_gender_match(t[0], str_cand), process.extract(str_cand, str_candidates.difference([str_cand]), limit=10))
  ret.extend(tuple(t[0].split()) for t in ocands)
  return ret
'''

def dfs(candidate_map, cand):
  visited = set([])
  s = deque([cand])
  while len(s) > 0:
    c = s.pop()
    visited.add(c)
    s.extend(filter(lambda c: c not in visited, candidate_map[c]))
  return visited.difference([cand])

if __name__ == '__main__':
  candidates = {
    ('Tom',): 1,
    ('Tom', 'Sawyer'): 1,
    ('Sid', 'Sawyer'): 1,
    ('Sally', 'Sawyer'): 1,
    ('Mr.', 'Tom'): 1,
    ('Mr.', 'Sawyer'): 1,
    ('Mrs.', 'Sawyer'): 1,
    ('Huck',): 1,
    ('Huckleberry',): 1,
    ('Huck', 'Finn'): 1,
    ('Huckleberry', 'Finn'): 1,
    ('Mr.', 'Finn'): 1,
    ('Sawyer',): 1,
    ('Sawyers',): 1,
    ('Fred', 'Weasley'): 1,
    ('Freddy', 'Weasley'): 1,
    ('Frederick', 'Weasley'): 1,
    ('Tweedledee',): 1,
    ('Tweedledum',): 1
  }

  import pprint
  pprint.pprint(disambiguate(candidates))
