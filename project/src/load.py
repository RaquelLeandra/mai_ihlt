from os import listdir, path as pth
import pandas as pd
import csv


def load_all(dir):
    files = listdir(dir)
    all_data = pd.DataFrame(columns=['sentence0','sentence1'])
    all_labels = pd.DataFrame(columns=['labels'])
    for file in files:
        path = pth.join(dir, file)
        if 'input' in path:  # Only read input files
            fd = pd.read_csv(path, sep='\t', lineterminator='\n', names=['sentence0', 'sentence1'], header=None,
                             quoting=csv.QUOTE_NONE)
            all_data = all_data.append(fd)
            fd = pd.read_csv(path.replace('input', 'gs'), sep='\t', lineterminator='\n', names=['labels'], header=None,
                             quoting=csv.QUOTE_NONE)
            all_labels = all_labels.append(fd)

    return all_data, all_labels