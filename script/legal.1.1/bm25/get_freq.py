import argparse
import cProfile
import io
import json
import logging
import math
import os
import pstats
import re
import string
import time
from datetime import date, datetime
from functools import wraps
from pathlib import Path
from string import punctuation
from tqdm import tqdm

import nltk
import numpy as np
from gensim import corpora
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

pattern = rf"[{punctuation}\s]+"
stopwords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def create_concat_text(doc_id_list, data_path):
    """Load documents and return concatenated document.

    Parameters
    ----------
    doc_id_list : list
        List of document IDs to load and concatenate together.
    data_path : str
        Path to load documents from.

    Returns
    -------
    doc_concat : str
        Concatenated string of advocate's cases.
    doc : dict
        Dictionary containing the processed text of each document.
    """

    docs = {}
    for doc_id in doc_id_list:
        flname = doc_id
        if (os.path.exists(os.path.join(data_path, f"{flname}.txt"))):
            with open(os.path.join(data_path, f"{flname}.txt"), 'r',encoding="utf8") as f:
                docs[flname] = process(f.read())

    # Concatenating into one document
    #  doc_concat = '\n'.join(docs.values())
    doc_concat = [token for doc in docs.values() for token in doc]
    #print(docs)
    #exit(0)
    return doc_concat, docs

def process(text):
    """Carry out processing of given text."""
    processed = list(filter(None, [re.sub('[^0-9a-zA-Z]+', '',
                                          token.lower())
                                   for token in re.split(pattern, text)]))

    # Removing tokens of length 1
    #processed = [lemmatizer.lemmatize(token) for token in processed if len(token) > 1 and token not in stopwords]

    return processed



def idf(corpus_size, doc_freq):
    """Calculate IDF given document frequency and number of documents.

    Parameters
    ----------
    corpus-size : int
        Number of documents.
    doc_freq : int
        Document frequency of token.

    Returns
    -------
    idf_score : float
        IDF score of the given token.
    """

    return math.log((corpus_size - doc_freq + 0.5)/(doc_freq + 0.5) + 1)



def convert_to_token_dict(corpus_dict, idx_dict):
    """Return dictionary after converting keys to appropriate format.

    Parameters
    ----------
    corpus_dict : dict
        Dictionary mapping IDs to other format.
    idx_dict : dict
        ID based dictionary to converting.

    Returns
    -------
    formatted_dict : dict
        Dictionary mapping tokens to their corresponding values.
    """
    formatted_dict = {
        corpus_dict[idx]: value
        for idx, value in iter(sorted(
            idx_dict.items(), key=lambda x: x[1], reverse=True))}

    return formatted_dict

def write_to_dir(obj, path, name, ext="json"):
    """Write object to path with given extenstion.

    Parameters
    ----------
    obj : object
        Object to be saved.
    path : str
        Path to save file.
    name : str
        Name to use for saving.
    ext : str, default "json"
        Extension to use for saving.
    """

    with open(os.path.join(path, f"{name}.{ext}"), 'w') as f:
        if (ext == 'json'):
            json.dump(obj, f, indent=4)
        else:
            for item in obj:
                print(item, file=f, end='\n')

def get_scores(freqs, per_doc_freqs, per_doc_lens, avg_doc_len,
               idfs, k1=1.5, b=0.75):
    """Calculate BM25 scores of a document against documents in the corpus.

    Parameters
    ----------
    freqs : dict
        Dictionary of tokens and frequencies of document.
    per_doc_freqs : dict
        Dictionary of frequencies of documents in the corpus.
    per_doc_lens : dict
        Length of documents in the corpus.
    avg_doc_len : float
        Average length of documents in the corpus.
    idfs : dict
        IDFs of all tokens.
    k1 : float, default=1.5
        Parameter for BM25.
    b : float, default=0.75
        Parameter for BM25.

    Returns
    -------
    scores : dict
        BM25 scores of documents.
    """

    adv_ordering = [*per_doc_freqs]
    doc_len_array = np.array([per_doc_lens[adv] for adv in adv_ordering])

    scores_array = np.zeros(len(adv_ordering))

    for token, freq in freqs.items():
        token_freq = np.array([per_doc_freqs[doc].get(token, 0)
                               for doc in adv_ordering])
        scores_array += compute_bm25(freq, idfs.get(token, 0),
                                     token_freq, doc_len_array, avg_doc_len,
                                     k1, b)
    scores = {adv: scores_array[i] for i, adv in enumerate(adv_ordering)}
    scores = {
        adv: score
        for adv, score in iter(sorted(
            scores.items(), key=lambda x: x[1], reverse=True))}

    return scores

def compute_bm25(freq, idf_score, token_freq,
                 doc_len_array, avg_doc_len, k1=1.5, b=0.75):
    """Return BM25 score of a token.

    Parameters
    ----------
    freq : int
        Frequency of token in test case.
    idf_score : float
        IDF score of token.
    token_freq : array_like
        Array of frequencies of token in corpus.
    doc_len_array : array_like
        Array of lengths of documents in corpus.
    avg_doc_len : float
        Average length of documents in corpus.
    k1 : float, default=1.5
        Parameter for BM25.
    b : float, default=0.75
        Parameter for BM25.

    Returns
    -------
    bm25_score : float
        BM25 score of token.
    """
    return ((idf_score * token_freq * (k1 + 1)
             / (token_freq + k1 * (1 - b + b * doc_len_array / avg_doc_len)))
            * freq)


def get_doc_freqs(doc_dict, corpus_dict, high_doc_freq_tokens):
    """Return dictionary containing the frequencies of tokens in documents

    Parameters
    ----------
    doc_dict : dict
        Dictionary containing the documents.
    corpus_dict : dict
        Dictionary to convert documents to bow.
    high_doc_freq_tokens : list
        Tokens to exclude.

    Returns
    -------
    dict_obj : dict
        Dictionary with the documents frequencies
    """

    dict_obj = {
        k: convert_to_token_dict(
            corpus_dict, {idx: v
                          for (idx, v) in corpus_dict.doc2bow(text)
                          if (corpus_dict[idx] not in high_doc_freq_tokens)})
        for k, text in doc_dict.items()}

    return dict_obj


def run_one_fold(foldnum,input_path,op,high_count_adv_split):
    """Compute Frequency stats of training documents of one fold.

    Parameters
    ----------
    foldnum : int
        Number of fold of n-cross validation.
    data_path : str
        Path to load json data from and to save generated dictionaries
    input_path : path to load data from
    """
    #input_path=r"train_cases_fold_0\fold_0"
    # Output path to store trained model
    # op="rerun"
    output_path = os.path.join(op,rf"fold_{foldnum}")
    bm25_path = os.path.join(op,rf"fold_{foldnum}")

    if not os.path.isdir(op):  # Check and make output directory
        os.makedirs(op)
    if not os.path.isdir(output_path):  # Check and make output directory
        os.makedirs(output_path)


    # Loading dictionary containing the 'train', 'test' and 'val' splits
    # of the high count advocates
    #rf"res\high_count_advs_{foldnum}.json"
    with open(
            high_count_adv_split,
            'r') as f:
        high_count_advs = json.load(f)

    adv_concat = {}  # For storing the concatenated advocate texts
    test_doc_ids = set()  # For storing the IDs of the test documents
    train_texts = {}  # For storing the training documents
    test_texts = {}  # For storing the test documents
    scores = {}  # For storing the BM25 scores
    test_doc_freqs = {}  # For storing the word frequencies of test documenst
    train_doc_freqs = {}  # For storing the word frequencies of training
    # documents

    # Creating the concatenated advocate representations to pass to the BM25
    # model
    for adv, cases in high_count_advs.items():

        # Creating the training corpus
        """for caseid in cases["train"]:
            with open(os.path.join(input_path, f"{caseid}.txt"), 'r',encoding="utf8") as f:
                adv_concat[caseid] = process(f.read())"""
        adv_concat[adv], docs = create_concat_text(cases["db"]+cases["train"], input_path)
        train_texts = {**train_texts, **docs}

        # Getting the list of test cases after removing prefixes
        test_doc_ids.update(map(lambda x: x, cases["test"]))
        test_doc_ids.update(map(lambda x: x, cases["val"]))
        #test_doc_ids.update(map(lambda x: x, cases["train"]))

    # Getting number of documents in the corpus_freqs
    corpus_size = len([*adv_concat])

    # Creating a dictionary for the dfs and cfs
    dictionary = corpora.Dictionary(adv_concat.values())

    # Getting all the frequencies needed for calculating the BM25 scores
    # Conversion from idx:value to token:value
    corpus_freqs = convert_to_token_dict(dictionary, dictionary.cfs)
    doc_freqs = convert_to_token_dict(dictionary, dictionary.dfs)

    # Percentage threshold for tokens to not consider in the BM25 scoring
    threshold = 0.70

    # Getting the list of tokens to not consider
    high_doc_freq_tokens = set(
        [token for token, freq in doc_freqs.items()
         if (freq >= threshold * corpus_size)])

    _ = [(corpus_freqs.pop(token), doc_freqs.pop(token)) for token in
         high_doc_freq_tokens]

    # Inverse documents frequencies
    inv_doc_freqs = {
        token: idf(corpus_size, value)
        for token, value in doc_freqs.items()}

    # Sorting
    inv_doc_freqs = {
        k: v
        for k, v in iter(sorted(
            inv_doc_freqs.items(), key=lambda x: x[1]))}

    # Per Document Frequencies
    per_doc_freqs = get_doc_freqs(
        adv_concat, dictionary, high_doc_freq_tokens)

    # Document Lengths
    per_doc_lens = {
        adv: sum(freqs.values())
        for adv, freqs in per_doc_freqs.items()}

    # Number of unique tokens
    unique_doc_lens = {
        adv: len(freqs.keys())
        for adv, freqs in per_doc_freqs.items()}

    # Average Document Length
    avg_doc_len = float(sum(per_doc_lens.values()))/corpus_size

    # Computing the BM25 scores for the test documents
    for idx in test_doc_ids:
        if (os.path.exists(os.path.join(input_path, f"{idx}.txt"))):
            with open(os.path.join(input_path, f"{idx}.txt"), 'r',encoding="utf8") as f:
                test_text = f.read()
            if (test_text == ''):
                continue
            test_texts[idx] = process(test_text)

    test_doc_freqs = get_doc_freqs(
        test_texts, dictionary, high_doc_freq_tokens)

    scores = {
        idx: get_scores(test_doc_freq, per_doc_freqs,
                        per_doc_lens,
                        avg_doc_len,
                        inv_doc_freqs)
        for idx, test_doc_freq in test_doc_freqs.items()}

    train_doc_freqs = get_doc_freqs(
        train_texts, dictionary, high_doc_freq_tokens)

    write_to_dir(doc_freqs, output_path, "doc_freqs")
    write_to_dir(corpus_freqs, output_path, "corpus_freqs")
    write_to_dir(inv_doc_freqs, output_path, "inv_doc_freqs")
    write_to_dir(per_doc_freqs, output_path, "per_doc_freqs")
    write_to_dir(per_doc_lens, output_path, "per_doc_lens")
    write_to_dir(unique_doc_lens, output_path, "unique_doc_lens")
    write_to_dir(scores, bm25_path, "scores")
    write_to_dir(test_doc_freqs, output_path, "test_doc_freqs")
    write_to_dir(train_doc_freqs, output_path, "train_doc_freqs")
    write_to_dir(high_doc_freq_tokens, output_path, "high_doc_freq_tokens",
                 "txt")
    dictionary.save(os.path.join(
        output_path, "dictionary"))


for i in range(5):
        run_one_fold(i)
        print(f"fold {i} done")