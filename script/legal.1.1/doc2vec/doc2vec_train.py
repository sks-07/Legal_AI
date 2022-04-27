

import argparse
import json
import logging
import multiprocessing as mp
import os
import pickle
import string

import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer



stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

pos_tag_dict = {
    'J': wordnet.ADJ,
    'V': wordnet.VERB,
    'N': wordnet.NOUN,
    'R': wordnet.ADV
}


# Methods


def get_wordnet_pos(treebank_tag):
    """Return wordnet POS tagging for better wordnet lemmatization

    Parameters
    ----------
    treebank_tag : str
        POS tag.

    Returns
    -------
    wordnet_tag : wordnet tag
        Corresponding wordnet tag of the parameter `treebank_tag`.
    """

    global pos_tag_dict
    return pos_tag_dict.get(treebank_tag[0], wordnet.NOUN)



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


def create_pd(split_dict, base_path):
    """Create Pandas Series from dictionary of IDs and base path

    Parameters
    ----------
    split_dict : dict
        Dictionary containing train, db, test and val splits of advocates
    base_path : str
        Base path where texts of the data are stored

    Returns
    -------
    df : pandas.Series
        Pandas Series containing the texts of the training documents

    Notes
    -----
    Takes a dictionary containing the train, test and validation document
    splits of each advocate and creates a Pandas Series object from the
    training documents. Applies pre-processing on the documents.
    See `process` for more information on pre-processing.
    

    train_doc_ids = set()
    train_docs_dict = {}

    _ = [train_doc_ids.update([
        idx for idx in [*cases["train"], *cases["db"]]])
        for adv, cases in split_dict.items()]

    # Loading the document texts into a dictionary
    for idx in train_doc_ids:
        if (os.path.exists(os.path.join(base_path, f"{idx}.txt"))):
            with open(os.path.join(base_path, f"{idx}.txt"), 'r',encoding="utf8") as f:
                train_docs_dict[idx] = f.read()
    """
    
    train_doc_ids=os.listdir(base_path)
    train_docs_dict = {}
    for idx in train_doc_ids:
        with open(os.path.join(base_path, idx), 'r',encoding="utf8") as f:
            train_docs_dict[idx] = f.read()

    df = pd.Series(train_docs_dict, name="FactText")
    df.index_name = "ID"
    df.reset_index()
    df = df.apply(process)
    return df



def run_one_fold(foldnum, epochs):
    """Train doc2vec on one fold of n-fold cross validation.

    Parameters
    ----------
    foldnum : int
        Number of fold.
    dict_path : str
        Base path to load high count advocates dictionary from.
    data_path : str
        Path to load data from.
    save_path : str
        Base path to save data.
    """

    dirs = os.path.join(r'eval')
    data_path=r'evaluate\fold_0\train'

    if not os.path.isdir(dirs):
        os.makedirs(dirs)

    #with open(rf"res\high_count_advs_{foldnum}.json", 'r') as f:
    #    high_count_advs = json.load(f)

    #db_cases = list(set([case for adv in high_count_advs.values()
    #                     for case in adv["train"]]))

    
    

    df = create_pd({}, data_path)
    tagged_docs = [TaggedDocument(txt, [idx]) for idx, txt in df.iteritems()]

    model = Doc2Vec(
        vector_size=200, epochs=epochs)
    model.build_vocab(tagged_docs)
    model.train(
        tagged_docs, total_examples=model.corpus_count,
        epochs=model.epochs)

    #model.dv.save(os.path.join(dirs, "d2v.docvectors"))
    #model.wv.save(os.path.join(dirs, "d2v.wordvectors"))
    model.save(os.path.join(dirs, "d2v.model"))
    
    """dv = model.dv
    db_dict = {}
    train_dict = {}
    for key in dv.index_to_key:
        if (key in db_cases):
            db_dict[key] = dv[key]
        else:
            train_dict[key] = dv[key]

    with open(os.path.join(dirs, "db_rep.pkl"), 'wb') as f:
        pickle.dump(db_dict, f)

    with open(os.path.join(dirs, "train_rep.pkl"), 'wb') as f:
        pickle.dump(train_dict, f)"""




#for i in range(5):
run_one_fold(0,20)
        #print(f"fold {i} done")