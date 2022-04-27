"""Takes a set of advocate representations and test case representations and
computes the cosine similarity between them and saves them."""

import json
import os
from datetime import date
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

# Date
date = date.today()
#  date = '2021-08-04'


def convert_to_ndarray(ordered_list, path):
    """Takes a list of files and a path and loads the data into a numpy
    array."""
    to_ndarray = []
    for name in ordered_list:
        to_ndarray.append(np.load(os.path.join(path,  name)))

    return np.stack(to_ndarray, axis=0)


def cosine(A, B):
    return np.dot(A, B) * 1./(np.linalg.norm(A)*np.linalg.norm(B))


def scale(matrix):
    row_max = np.reshape(np.amax(matrix, axis=-1), (matrix.shape[0], 1))
    row_min = np.reshape(np.amin(matrix, axis=-1), (matrix.shape[0], 1))
    scaled = (matrix - row_min) * 1./(row_max - row_min)
    return scaled


def main(foldnum):

    # Base path
    

    # Save path
    save_path = rf"eval"

    # Advocates Path
    adv_path = rf"eval\advocates"

    # Test Cases
    test_cases_path = rf"eval\test_docs"

    # Getting all the test case representations together
    ordered_cases = os.listdir(test_cases_path)
    ndarray_cases = convert_to_ndarray(ordered_cases, test_cases_path)

    # Getting all the advocate representations together
    ordered_adv = os.listdir(adv_path)
    ndarray_adv = convert_to_ndarray(ordered_adv, adv_path)

    # Computing all cosine similarities
    cosine_sim = cdist(ndarray_cases, ndarray_adv, cosine)

    #  cosine_sim = scale(cosine_sim)

    scores = {}

    for i, case in enumerate(ordered_cases):
        score = {os.path.splitext(ordered_adv[j])[0]: val for j, val in
                 enumerate(cosine_sim[i, :])}
        scores[os.path.splitext(case)[0]] = score

    # Sorting the values in descending order for better human readability
    for key in scores:
        scores[key] = {k: v for k, v in sorted(scores[key].items(), key=lambda
                                               x: x[1], reverse=True)}

    if not(os.path.isdir(save_path)):
        os.makedirs(save_path)
    with open(os.path.join(save_path, f"sim_scores_{foldnum}.json"), 'w') as f:
        json.dump(scores, f, indent=4)



for i in range(5):
        main(i)
        print(f"fold {i} done")