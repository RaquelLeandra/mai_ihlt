from math import sqrt
from scipy.stats import pearsonr

def cosine_dist(setA, setB):
    return len(setA.intersection(setB)) / sqrt(len(setA) * len(setB))

def jaccard_dist(setA, setB):
    return len(setA.intersection(setB)) / len(setA.union(setB))

def overlap_dist(setA, setB):
    return len(setA.intersection(setB)) / min([len(setA), len(setB)])

def dice_dist(setA, setB):
    return 2 * len(setA.intersection(setB)) / (len(setA) + len(setB))

def cosine_sim(vectorA, vectorB):
    if type(vectorA) != set: vectorA = set(vectorA)
    if type(vectorB) != set: vectorB = set(vectorB)
    return 1. - cosine_dist(vectorA, vectorB)

def jaccard_sim(vectorA, vectorB):
    if type(vectorA) != set: vectorA = set(vectorA)
    if type(vectorB) != set: vectorB = set(vectorB)
    return 1. - jaccard_dist(vectorA, vectorB)

def overlap_sim(vectorA, vectorB):
    if type(vectorA) != set: vectorA = set(vectorA)
    if type(vectorB) != set: vectorB = set(vectorB)
    return 1. - overlap_dist(vectorA, vectorB)

def dice_sim(vectorA, vectorB):
    if type(vectorA) != set: vectorA = set(vectorA)
    if type(vectorB) != set: vectorB = set(vectorB)
    return 1. - dice_dist(vectorA, vectorB)

def sim_stats_dataframe(dataframe, gs_dataframe):
    sim_functions = {
        'jaccard': jaccard_sim,
        'cosine ': cosine_sim,
        'overlap': overlap_sim,
        'dice   ': dice_sim
    }
    for name, sim_func in sim_functions.items():
        sims_guess = []
        for i in dataframe.index:
            sentenceA = dataframe.loc[i, 'sentence0']
            sentenceB = dataframe.loc[i, 'sentence1']
            sims_guess.append(sim_func(sentenceA, sentenceB))
        print('Pearson correlation (using ' + name + '):', pearsonr(gs_dataframe['labels'], sims_guess)[0])

