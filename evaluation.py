from labeling import *
from disambiguation import *
import networkx as nx

def evaludate_candidates(characters, candidates):
    (matching, G) = match_candidates_and_characters(characters, candidates)
    unresolved_characters = []
    duplicate_candidates = []
    invalid_candidates = []

    for character in characters:
        if character not in matching:
            unresolved_characters.append(character)

    for cand in candidates:
        if cand not in matching:
            if G.degree(cand) == 0:
                invalid_candidates.append(cand)
            else:
                duplicate_candidates.append(cand)

    return (unresolved_characters, duplicate_candidates, invalid_candidates)

if __name__ == '__main__':
    candidates = {
    ('Tom',): 1,
    ('Tom', 'Sawyer'): 1,
    ('Sid', 'Sawyer'): 1,
    ('Sally', 'Sawyer'): 1,
    ('Mr.', 'Tom'): 1,
    ('Mr.', 'Sawyer'): 1,
    ('Mrs.', 'Sawyer'): 1,
    ('T.', 'Sawyer'): 1,
    ('Doctor', 'Sawyer'): 1,
    ('Monsieur', 'Sawyer'): 1,
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
    ('Tweedledum',): 1,
    ('Monsieur',): 1,
    ('D\'Artagan',): 1,
    ('d\'Artagan',): 1
    }

    candidates = find_unique_characters(candidates)
    characters = {}  
    get_sparknote_characters_from_file('huckfinn', characters)
    
    (unresolved_characters, duplicate_candidates, invalid_candidates) = evaludate_candidates(characters, candidates)
    print unresolved_characters
    print duplicate_candidates
    print invalid_candidates
