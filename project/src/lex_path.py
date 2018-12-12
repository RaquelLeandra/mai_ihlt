from nltk.corpus import wordnet as wn


def genData(word):
    hyper   = lambda s: s.hypernyms()
    synset  = wn.synset(word)
    tree    = synset.tree(hyper)
    closure = synset.closure(hyper)
    return tree, list(closure)


def listIntersec(list1, list2):
    return list(set(list1).intersection(set(list2)))


def pathTo(synset, tree, path=None):
    if path is None:
        path = []

    path += [tree[0]]

    if tree[0] == synset:
        return True, path  # Found

    for i in range(1, len(tree)):
        found, path = pathTo(synset, tree[i], path)
        if found: return True, path  # Found in child

    return False, path  # Not found


def lex_compare(A, B):
    treeA, closureA = genData(A)
    treeB, closureB = genData(B)
    closureA += [treeA[0]]  # Include the root in the closure
    closureB += [treeB[0]]  # Include the root in the closure
    common_hyper = listIntersec(closureA, closureB)  # Find which common hypernyms they both have in common

    min_dist = float('+inf')
    min_common = None
    min_pathA = []
    min_pathB = []

    for i in common_hyper:  # For each common hypernym, find the path from the root to the hypernym for both synsets
        pathA = pathTo(i, treeA, [])[1]
        pathB = pathTo(i, treeB, [])[1]
        dist = len(pathA) + len(pathB) - 2
        if dist < min_dist:  # If the total path is better than the one we had before, update it
            min_dist = dist
            min_common = i
            min_pathA = pathA
            min_pathB = pathB

    full_path = min_pathA[:-1] + min_pathB[::-1]  # Remove the repeated common word
    return full_path, min_common, min_dist
