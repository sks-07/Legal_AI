"""
Script that generates the advocate representations. It does this by taking the 'train' documents for an advocate and concatenating them together to create an advocate 'train' document on which doc2vec is run. doc2vec is similarly run on the 'test' documents to get their representations.
"""

import json
import os
import string
from datetime import date
from pathlib import Path
import pickle
import numpy as np
from gensim.models.doc2vec import Doc2Vec
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

from helpers import get_wordnet_pos

# Date
date = date.today()
#  date = '2021-08-04'
stop_words = set(stopwords.words('english'))
def process(text):
    """Carry out stopword removal, stripping and lemmatization of text.

    Parameters
    ----------
    text : str
        Text to process.

    Returns
    -------
    tokenized_text : str
        Processed text.
    """

    tokenized_text = word_tokenize(text.lower())
    tokenized_text = [token.strip(string.punctuation) for token in
                      tokenized_text]

    # Stopword and punctuation removal
    tokenized_text = list(filter(
        lambda token: (token not in string.punctuation and
                       token not in stop_words),
        tokenized_text))

    # Lemmatization POS tags
    #pos_tags = pos_tag(tokenized_text)

    #tokenized_text = [lemmatizer.lemmatize(token, get_wordnet_pos(pos_tag))  for (token, pos_tag) in pos_tags]

    #  tokenized_text = " ".join(tokenized_text)

    return tokenized_text
    
def create_concat_text(doc_id_list, data_path):
    """Takes a given path and set of document ids and returns the texts of the
    given ids from the path concatenated into one document."""
    docs = []
    for doc_id in doc_id_list:
        flname = doc_id
        if(os.path.exists(os.path.join(data_path, f"{flname}.txt"))):
            with open(os.path.join(data_path, f"{flname}.txt"), 'r',encoding="utf8") as f:
                docs.append(f.read())

    # Concatenating into one document
    #doc_concat = '\n'.join(docs)###
    #return doc_concat###
    return docs


def remove_prefix(doc_id_list):
    """Takes a list of document IDs, removes prefixes and returns them without
    duplicates."""
    return list(set(map(lambda x: x[2:], doc_id_list)))


def create_reps(doc_id_list, data_path, o_path, model):
    """Given a list of document IDs, a data path, an output path and a model,
    creates the vector representations of the documents using the model and
    saves them to the output path."""

    for doc_id in tqdm(doc_id_list):
        with open(os.path.join(data_path, doc_id), 'r',encoding="utf8") as f:
                # Processing each test case
            doc_processed = process(f.read())

        # Generating the doc2vec representation of the test case
        doc_vec = model.infer_vector(doc_processed, epochs=10)

        # Saving the test case with the advocate name as a prefix
        #np.save(os.path.join(o_path, f"{doc_id}.npy"),
        #        doc_vec)

        with open(os.path.join(o_path, f"{doc_id.split('.')[0]}.pkl"), 'wb') as f:
            pickle.dump(doc_vec, f)


def main(foldnum):
    # Base path for Delhi High Court advocate cases
    path = os.path.join(os.path.expanduser("~"), "Datasets", "DelhiHighCourt")

    # Path to store results
    output_path = os.path.join(r'eval', f"fold_{foldnum}")
    if not os.path.isdir(output_path):  # Check and make output directory
        os.makedirs(output_path)

    # Path for high count advocates json
    high_count_path = rf"res\high_count_advs_{foldnum}.json"

    # Path to textual data
    data_path = r"facts"

    # Loading the pre-trained model
    model = Doc2Vec.load(os.path.join(os.path.join(r'rerun\setup2\doc2vec', f"fold_{foldnum}"), "d2v.model"))

    # For getting the train documents for the high count advocates
    with open(rf"D:\Thesis\New folder\Resources\high_count_advs_{foldnum}.json", 'r') as f:
        high_count_advs = json.load(f)

    # Creating directory for advocates if it does not exist
    if not(os.path.isdir(os.path.join(output_path, "advocates"))):
        os.makedirs(os.path.join(output_path, "advocates"))

    # Making the directory for storing the document representations of the
    # test documents if it does not exist
    if not(os.path.isdir(os.path.join(output_path, "test_docs"))):
        os.makedirs(os.path.join(output_path, "test_docs"))

    # Making the directory for storing the document representations of the
    # val documents if it does not exist
    #if not(os.path.isdir(os.path.join(output_path, "val_docs"))):
    #    os.makedirs(os.path.join(output_path, "val_docs"))

    for key, cases in tqdm(high_count_advs.items()):
        # Concatenating all the train documents together
        train_list = cases['train']+cases['db']
        train_concat = create_concat_text(train_list, data_path)#

        # Processing the concatenated document
        train_processed = process(train_concat)#

        # Using the trained doc2vec model to get an advocate representation
        train_vec = model.infer_vector(train_processed, epochs=5)#


        ##create_reps(train_list, data_path, os.path.join(output_path,"advocates"), model)
        # Saving the representation
        np.save(os.path.join(output_path,"advocates", f"{key}.npy"), train_vec)#

        # Removing any duplicate test cases in the instance more than one high count advocate worked on the same case
        test_doc_list = cases['test']+cases['val']
        #val_doc_list = cases['val']

        # Generating the vector representations and saving them
        create_reps(test_doc_list, data_path, os.path.join(output_path,
                                                           "test_docs"), model)

        #create_reps(val_doc_list, data_path, os.path.join(output_path,"val_docs"), model)


def mainrun():
    output_path = os.path.join(r'eval\pkl_vec')
    if not os.path.isdir(output_path):  # Check and make output directory
        os.makedirs(output_path)

    
    # Path to textual data
    data_path = r'evaluate\fold_0'

    # Loading the pre-trained model
    model = Doc2Vec.load(os.path.join('eval', "d2v.model"))

    # Creating directory for advocates if it does not exist
    if not(os.path.isdir(os.path.join(output_path, "advocates"))):
        os.makedirs(os.path.join(output_path, "advocates"))

    # Making the directory for storing the document representations of the
    # test documents if it does not exist
    if not(os.path.isdir(os.path.join(output_path, "test_docs"))):
        os.makedirs(os.path.join(output_path, "test_docs"))

    # Making the directory for storing the document representations of the
    # val documents if it does not exist
    #if not(os.path.isdir(os.path.join(output_path, "val_docs"))):
    #    os.makedirs(os.path.join(output_path, "val_docs"))


    train_doc_ids=os.listdir(os.path.join(data_path,'train'))
    test_doc_ids=os.listdir(os.path.join(data_path,'test'))
    create_reps(train_doc_ids, os.path.join(data_path,'train'), os.path.join(output_path,"advocates"), model)
    create_reps(test_doc_ids, os.path.join(data_path,'test'), os.path.join(output_path,"test_docs"), model)

    


#for i in range(5):
#    main(i)
#    print(f"fold {i} done")

mainrun()