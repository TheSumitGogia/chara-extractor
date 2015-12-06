from labeling import *
import networkx as nx

def evaludate_candidates(characters, candidates):
    (matching, G) = match_candidates_and_characters(characters, candidates)
    unresolved_characters = []
    duplicate_candidates = []
    invalid_candidates = []

    for character in characters:
        if character not in mactching:
            unresolved_characters.append(character)

    for cand in candidates:
        if cand not in matching:
            if G.degree(cand) == 0:
                invalid_candidates.append(cand)
            else:
                duplicate_candidates.append(cand)

    return (resolved_characters, duplicate_candidates, invalid_candidates)
