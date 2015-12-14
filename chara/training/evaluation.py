from chara.labeling.labeler import *
from chara.resolve.disambiguation import *
import networkx as nx
import matplotlib.pyplot as plt

def evaluate_candidates(characters, candidates):
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
    candidates = [('Dr.', 'Van', 'Helsing'), ('Lord', 'Godalming'), ('Jonathan', 'Harker'), ('Dr.', 'Seward'), ('Count',), ('Lucy', 'Westenra'), ('Mr.', 'Hawkins'), ('Renfield',), ('Arthur',), ('Mr.', 'Quincey'), ('Mrs.', 'Harker'), ('Quincey', 'Morris'), ('friend', 'John'), ('Professor',), ('Mina', 'Murray'), ('Mr.', 'Morris')]

    print candidates
    candidates = find_unique_characters(candidates)
    print candidates
    characters = {}
    get_sparknote_characters_from_file('dracula', characters)

    (unresolved_characters, duplicate_candidates, invalid_candidates) = evaluate_candidates(characters, candidates)
    print unresolved_characters
    print duplicate_candidates
    print invalid_candidates
