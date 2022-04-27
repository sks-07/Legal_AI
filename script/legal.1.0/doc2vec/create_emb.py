"""
Script that generates the advocate representations. It does this by taking the 'train' documents for an advocate and concatenating them together to create an advocate 'train' document on which doc2vec is run. doc2vec is similarly run on the 'test' documents to get their representations.
"""

import json
import os
import string
from datetime import date
from pathlib import Path

import numpy as np
from gensim.models.doc2vec import Doc2Vec
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

from helpers import get_wordnet_pos, process

# Date
date = date.today()
#  date = '2021-08-04'


def create_concat_text(doc_id_list, data_path):
    """Takes a given path and set of document ids and returns the texts of the
    given ids from the path concatenated into one document."""
    docs = []
    for doc_id in doc_id_list:
        flname = doc_id[2:]
        if(os.path.exists(os.path.join(data_path, f"{flname}.txt"))):
            with open(os.path.join(data_path, f"{flname}.txt"), 'r') as f:
                docs.append(f.read())

    # Concatenating into one document
    doc_concat = '\n'.join(docs)

    return doc_concat


def remove_prefix(doc_id_list):
    """Takes a list of document IDs, removes prefixes and returns them without
    duplicates."""
    return list(set(map(lambda x: x[2:], doc_id_list)))


def create_reps(doc_id_list, data_path, o_path, model):
    """Given a list of document IDs, a data path, an output path and a model,
    creates the vector representations of the documents using the model and
    saves them to the output path."""

    for doc_id in tqdm(doc_id_list):
        with open(os.path.join(data_path, f"{doc_id}.txt"), 'r') as f:
            # Processing each test case
            doc_processed = process(f.read())

        # Generating the doc2vec representation of the test case
        doc_vec = model.infer_vector(doc_processed, epochs=10)

        # Saving the test case with the advocate name as a prefix
        np.save(os.path.join(o_path, f"{doc_id}.npy"),
                doc_vec)


def main():
    # Base path for Delhi High Court advocate cases
    path = os.path.join(os.path.expanduser("~"), "Datasets", "DelhiHighCourt")

    # Path to store results
    output_path = "./re/Result/newadvocates/"

    # Path for high count advocates json
    high_count_path = "./Resource/20_fold/fold_0/high_count_advs_0.json"

    # Path to textual data
    data_path = "./Resource/facts/"
    # Loading the pre-trained model
    model = Doc2Vec.load("./re/Result/d2v.model")

    # For getting the train documents for the high count advocates
    with open(high_count_path, 'r') as f:
        high_count_advs = json.load(f)

    # Creating directory for advocates if it does not exist
    if not(os.path.isdir(os.path.join(output_path, "newadvocates"))):
        os.makedirs(os.path.join(output_path, "newadvocates"))

    # Making the directory for storing the document representations of the
    # test documents if it does not exist
    if not(os.path.isdir(os.path.join(output_path, "train_docs"))):
        os.makedirs(os.path.join(output_path, "train_docs"))

    if not(os.path.isdir(os.path.join(output_path, "test_docs"))):
        os.makedirs(os.path.join(output_path, "test_docs"))

    # Making the directory for storing the document representations of the
    # val documents if it does not exist
    if not(os.path.isdir(os.path.join(output_path, "val_docs"))):
        os.makedirs(os.path.join(output_path, "val_docs"))

    for key, cases in tqdm(high_count_advs.items()):
        # Concatenating all the train documents together
        train_list = cases['train']
        train_concat = create_concat_text(train_list, data_path)

        # Processing the concatenated document
        train_processed = process(train_concat)

        # Using the trained doc2vec model to get an advocate representation
        train_vec = model.infer_vector(train_processed, epochs=5)

        # Saving the representation
        np.save(os.path.join(output_path,
                             "advocates", f"{key}.npy"), train_vec)

        # Removing any duplicate test cases in the instance more than one high count advocate worked on the same case
        test_doc_list = cases['test']
        val_doc_list = cases['val']

        # Generating the vector representations and saving them
        create_reps(train_list, data_path, os.path.join(output_path,
                                                          "train_docs"), model)

        create_reps(test_doc_list, data_path, os.path.join(output_path,
                                                           "test_docs"), model)

        create_reps(val_doc_list, data_path, os.path.join(output_path,
                                                          "val_docs"), model)


if __name__ == '__main__':
    import cProfile
    import pstats
    import io
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s).sort_stats('cumtime')
    stats.print_stats()

    with open(Path(
            f"./create_rep_run_stats.log"),
            'w+') as f:
        f.write(s.getvalue())