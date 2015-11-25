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
    refs = partial_reference(candidates, cand) + \
           nickname_resolution(candidates, cand) + \
           title_resolution(candidates, cand)
    all_maps[cand] = set(refs)

  return all_maps

def contains_tuple(t_outer, t_inner):
  for i in range(len(t_outer) - len(t_inner) + 1):
    if t_outer[i:i + len(t_inner)] == t_inner:
      return True
  return False

def fuzzy_contains_tuple(t_outer, t_inner):
  for i in range(len(t_outer) - len(t_inner) + 1):
    if all(a in b for a, b in zip(t_inner, t_outer[i:i + len(t_inner)])):
      return True
  return False

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

def title_resolution(candidates, cand):
  ret = []
  for ocand in candidates:
    if cand != ocand and cand[0] in ALL_TITLES and cand[-1] == ocand[-1]:
      first_name = ocand[0].lower()
      if first_name in gender_dict:
        if gender_dict[first_name] == 'MALE' and cand[0] in MALE_TITLES:
          ret.append(ocand)
        elif gender_dict[first_name] == 'FEMALE' and cand[0] in FEMALE_TITLES:
          ret.append(ocand)
  return ret

if __name__ == '__main__':
  candidates = {
    ('Tom',): 1,
    ('Tom', 'Sawyer'): 1,
    ('Mr.', 'Tom'): 1,
    ('Mr.', 'Sawyer'): 1,
    ('Mrs.', 'Sawyer'): 1,
    ('Huck',): 1,
    ('Huckleberry',): 1,
    ('Huck', 'Finn'): 1,
    ('Huckleberry', 'Finn'): 1,
    ('Mr.', 'Finn'): 1
  }

  import pprint
  pprint.pprint(disambiguate(candidates))
