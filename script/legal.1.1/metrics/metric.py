"""Takes a path containing a set of targets and a path containing a set of
predictions and calculates the mean Average precision, the per-class precision
and recall."""
import argparse
import json
import multiprocessing as mp
import os
import time
from datetime import datetime
from functools import wraps
import pickle
import numpy as np

# For getting more information about numpy errors
np.seterr('raise')




# Decorators



def per_query_prec_rec(y_true, y_pred):
    """Takes a set of true values and their predictions and returns the
    precision and recall of each query .
    Shapes:
    y_true: (num_test, num_adv)
    y_pred: (num_test, num_adv)
    """
    # For storing the true and false positives and negatives of each class
    per_query_prec = []
    per_query_rec = []

    num_queries = y_true.shape[0]

    # Getting the positives and negatives of each class
    for query in range(num_queries):
        tp = np.dot(y_true[query, :], y_pred[query, :])
        pp = np.sum(y_pred[query, :])
        p = np.sum(y_true[query, :])

        prec = tp * 1./pp
        rec = tp * 1./p

        per_query_prec.append(prec)
        per_query_rec.append(rec)

    return per_query_prec, per_query_rec



def per_class_prec_rec(y_true, y_pred):
    """Takes a set of true values and their predictions and returns the
    precision and recall of each class(adv)"""
    per_class_prec = []
    per_class_rec = []

    num_classes = y_true.shape[-1]

    for cls in range(num_classes):
        tp = np.dot(y_true[:, cls], y_pred[:, cls])
        pp = np.sum(y_pred[:, cls])
        p = np.sum(y_true[:, cls])
        prec = tp/pp if pp != 0 else 0
        rec = tp/p if p != 0 else 0

        per_class_prec.append(prec)
        per_class_rec.append(rec)

    return per_class_prec, per_class_rec



def one_query_ap(precisions, y_true, relevance):
    """Computes the average precision of one query given the precision and
    relevance values."""
    return np.dot(
        precisions, relevance[:len(precisions)])*1./np.sum(y_true)


def one_class_ap(precisions, recalls):
    """Computes the average precision of one class given the precision and
    recall values."""
    return np.sum((recalls[1:] - recalls[:-1]) * precisions[:-1])



def mAP(ap_values):
    """Computes the mean average precision from the per-class average precision
    values."""
    return np.mean(ap_values)



def convert_to_ndarray(ordered_list, path):
    """Takes a list of files and a path and loads the data into a numpy
    array."""
    to_ndarray = []
    for name in ordered_list:
        to_ndarray.append(np.load(os.path.join(path,  name)))

    return np.stack(to_ndarray, axis=0)



def vectorize_prediction(scores, adv_index, k):
    """Takes a set of documents, the ranking of their retrieval items and
    a k value and vectorizes them."""
    vectorized_list = []
    for case in list(scores.keys()):
        # Getting the top k elements
        ranked_list = scores[case][:k]
        vector = np.zeros(shape=(len(adv_index.keys()),))
        for ranked_item in ranked_list:
            vector[adv_index[ranked_item]] = 1
        vectorized_list.append(vector)

    return np.stack(vectorized_list, axis=0)



def relevance_at_k(scores, adv_index, y_true):
    vectorized_list = []
    for i, case in enumerate(list(scores.keys())):
        ranked_list = scores[case]
        vector = np.zeros(shape=(len(ranked_list),))
        for j, ranked_item in enumerate(ranked_list):
            if(y_true[i, adv_index[ranked_item]] == 1):
                vector[j] = 1
        vectorized_list.append(vector)

    return np.stack(vectorized_list, axis=0)



def numpy_to_dict(array, cases, metric='P'):
    """Converts a numpy of precision or recall values into a dict"""
    numpy_dict = {}
    for i, case in enumerate(cases):
        numpy_dict[case] = {f"{metric}@{j+1}": value for j,
                            value in enumerate(array[i, :])}

    return numpy_dict


def numpy_to_dict_classes(array, adv_index, metric='P'):
    """Converts a numpy of precision or recall values into a dict"""
    numpy_dict = {}
    for adv, idx in adv_index.items():
        numpy_dict[adv] = {f"{metric}@{j+1}": value for j,
                           value in enumerate(array[idx, :])}

    return numpy_dict


def create_targets(targets_dict, adv_index, cases):
    """Create targets from a dictionary of targets and advocate ordering.

    Parameters
    ----------

    targets_dict : dict
        Dictionary with the targets of each case.
    adv_list : list
        List of advocates to consider.
    cases : list
        Ordered list cases.

    Returns
    -------
    result : numpy.ndarray
        Stacked target mult-hot vectors.
    """
    results = []
    for case in cases:
        results.append(np.array([int(adv in targets_dict[case])
                                 for adv in list(adv_index.keys())],
                                dtype=np.float32))

    return np.stack(results, axis=0)



def run_one_fold(numfold):
    """Carries out the metrics computation one one fold of an n-fold
    cross-validation setup."""

    #base_path = os.path.join(rf"eval\setup3", f"fold_{numfold}")
    base_path = r'D:\Thesis\Legal_AI\script\legal.1.1\metric\Han'
    if not(os.path.exists(base_path)):
        os.makedirs(base_path)
    
    with open(r"D:\Thesis\Legal_AI\script\legal.1.1\metric\Han\similarity_reranking.json", 'r') as f:
        scores = json.load(f)
    
    
    output_path = os.path.join(base_path, "metrics")
    if not(os.path.exists(output_path)):
        os.makedirs(output_path)

    with open("adv_list",'rb') as f:
        adv_list=pickle.load(f)
    
    

    adv_index = {k: i for i, k in enumerate(adv_list)}

    with open(r'D:\Thesis\Legal_AI\script\legal.1.1\IPC_data\case_targets.json', 'r') as f:
        targets = json.load(f)

    # Creating the ndarray to actual targets
    ndarray_true = create_targets(targets, adv_index, list(scores.keys()))

    # Ordering the scores for each advocate
    scores = {case_id: [adv for adv, score in sorted(pred.items(),
                                                     key=lambda x: x[1],
                                                     reverse=True)] for
              case_id, pred in scores.items()}

    # Top K
    top_k = np.arange(start=1, stop=10 + 1)

    # For storing the precision and recall scores across different thresholds
    precision_scores = []
    recall_scores = []

    per_class_precision_scores = []
    per_class_recall_scores = []

    ap_scores = []
    ap_dict = {}

    per_class_ap_scores = []
    per_class_ap_dict = {}

    # Computing relevance for AP calculation
    relevance = relevance_at_k(scores, adv_index, ndarray_true)

    for k in top_k:
        ndarray_pred = vectorize_prediction(scores, adv_index, k)

        # For each query
        prec, rec = per_query_prec_rec(ndarray_true, ndarray_pred)
        precision_scores.append(prec)
        recall_scores.append(rec)

        # For each advocate
        prec, rec = per_class_prec_rec(ndarray_true, ndarray_pred)
        per_class_precision_scores.append(prec)
        per_class_recall_scores.append(rec)

    # Stacking along first axis Shape = (top_k, num_queries)
    precision_scores = np.stack(precision_scores, axis=0)
    recall_scores = np.stack(recall_scores, axis=0)

    # Stacking along first axis Shape = (top_k, num_classes)
    per_class_precision_scores = np.stack(
        per_class_precision_scores, axis=0)
    per_class_recall_scores = np.stack(per_class_recall_scores, axis=0)

    # Shape = (num_queries, top_k)
    precision_scores = precision_scores.T
    recall_scores = recall_scores.T

    # Shape = (num_classes, top_k)
    per_class_precision_scores = per_class_precision_scores.T
    per_class_recall_scores = per_class_recall_scores.T

    # Calculating the AP scores for each query
    for query, case_id in enumerate(list(scores.keys())):
        ap = one_query_ap(precision_scores[query, :],
                          ndarray_true[query, :],
                          relevance[query, :])

        ap_dict[case_id] = ap
        ap_scores.append(ap)

    # Calculating the query mAP
    mean_ap = mAP(ap_scores)

    # Sorting the AP scores in descending order
    ap_dict = {k: v for k, v in sorted(ap_dict.items(), key=lambda x:
                                       x[1], reverse=True)}

    # Converting to a dictionary for human readability
    prec_dict = numpy_to_dict(precision_scores, list(scores.keys()), 'P')
    rec_dict = numpy_to_dict(recall_scores, list(scores.keys()), 'R')

    # Calculating the AP scores for each advocate
    for adv, idx in adv_index.items():
        ap = one_class_ap(per_class_precision_scores[idx, :],
                          per_class_recall_scores[idx, :])

        per_class_ap_dict[adv] = ap
        per_class_ap_scores.append(ap)

    # Calculating the class mAP
    per_class_mean_ap = mAP(per_class_ap_scores)

    # Sorting the AP scores in descending order
    per_class_ap_dict = {k: v for k, v in sorted(per_class_ap_dict.items(),
                                                 key=lambda x: x[1],
                                                 reverse=True)}

    # Converting to a dictionary for human readability
    per_class_prec_dict = numpy_to_dict_classes(per_class_precision_scores,
                                                adv_index, 'P')
    per_class_rec_dict = numpy_to_dict_classes(per_class_recall_scores,
                                               adv_index, 'R')

    # Saving all the generated data
    with open(os.path.join(output_path, "per_query_precision.json"),
              'w+') as f:
        json.dump(prec_dict, f, indent=4)

    with open(os.path.join(output_path, "per_query_recall.json"),
              'w+') as f:
        json.dump(rec_dict, f, indent=4)

    with open(os.path.join(output_path, "per_query_ap.json"),
              'w+') as f:
        json.dump(ap_dict, f, indent=4)

    with open(os.path.join(output_path, "mAP.txt"),
              'w+') as f:
        f.write(str(mean_ap))

    with open(os.path.join(output_path, "per_class_precision.json"),
              'w+') as f:
        json.dump(per_class_prec_dict, f, indent=4)

    with open(os.path.join(output_path, "per_class_recall.json"), 'w+') as f:
        json.dump(per_class_rec_dict, f, indent=4)

    with open(os.path.join(output_path, "per_class_ap.json"), 'w+') as f:
        json.dump(per_class_ap_dict, f, indent=4)

    with open(os.path.join(output_path, "class_mAP.txt"), 'a+') as f:
        f.write(str(per_class_mean_ap))

    np.save(os.path.join(output_path, "top_k"), top_k)
    np.save(os.path.join(output_path, "precision"), precision_scores)
    np.save(os.path.join(output_path, "recall"), recall_scores)
    np.save(os.path.join(output_path, "ap"), ap_scores)
    np.save(os.path.join(output_path, "per_class_precision"),
            per_class_precision_scores)
    np.save(os.path.join(output_path, "per_class_recall"),
            per_class_recall_scores)
    np.save(os.path.join(output_path, "per_class_ap"), per_class_ap_scores)

    print(f"fold {numfold}: mAP ={mean_ap}")


for i in range(5):
    run_one_fold(i)


#run_one_fold(0)       
