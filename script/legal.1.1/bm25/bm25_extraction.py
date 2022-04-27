import argparse
import cProfile
import io
import json
import logging
import math
import os
import pstats
import re
import pickle
import string
import time
from datetime import date, datetime
from functools import wraps
from pathlib import Path
from string import punctuation
from sqlalchemy import true
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import nltk
import numpy as np
from gensim import corpora
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

pattern = rf"[{punctuation}\s]+"
stopwords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def process(text):
    """Carry out processing of given text."""
    processed = list(filter(None, [re.sub('[^0-9a-zA-Z]+', '',
                                          token.lower())
                                   for token in re.split(pattern, text)]))

    # Removing tokens of length 1
    #processed = [lemmatizer.lemmatize(token) for token in processed if len(token) > 1 and token not in stopwords]

    return processed

def run_one_fold(foldnum,args):
    input_path=r"facts"
    # Output path to store trained model
    bm=rf'rerun\setup2\bm25'

    output_path = os.path.join(bm,f"fold_{foldnum}")
    bm25_path = os.path.join(bm,f"fold_{foldnum}")

    if not os.path.isdir(bm):  # Check and make output directory
        os.makedirs(bm)

    if not os.path.isdir(output_path):  # Check and make output directory
        os.makedirs(output_path)

    

    # Loading dictionary containing the 'train', 'test' and 'val' splits
    # of the high count advocates
    with open(
            rf"res\high_count_advs_{foldnum}.json",
            'r') as f:
        high_count_advs = json.load(f)

    adv_concat={}
    test_doc_ids = set()
    li=[]
    
    for adv,case in high_count_advs.items():
        for caseid in case['db']:
            if(os.path.exists(os.path.join(input_path, f"{caseid}.txt"))):
                with open(os.path.join(input_path, f"{caseid}.txt"), 'r',encoding="utf8") as f:
                    adv_concat[caseid] = process(f.read())
                    li.append(caseid)
        for caseid in case['train']:
            if(os.path.exists(os.path.join(input_path, f"{caseid}.txt"))):
                with open(os.path.join(input_path, f"{caseid}.txt"), 'r',encoding="utf8") as f:
                    adv_concat[caseid] = process(f.read())
                    li.append(caseid)
        
        test_doc_ids.update(map(lambda x: x, case["test"]+case["val"]))
        
    
    bm_25obj = BM25Okapi(adv_concat.values())
    pickle.dump(bm_25obj, open(os.path.join(output_path,"bm25obj.sav"), 'wb'))
    test_doc_id=[]
    for k in test_doc_ids:
        test_doc_id.append(k)
    
    """batch_size=20;
    for i in tqdm(range(0,len(test_doc_id),batch_size)):
        result={}
        for caseid in tqdm(test_doc_id[i:i+batch_size]):
            if(os.path.exists(os.path.join(input_path, f"{caseid}.txt"))):
                with open(os.path.join(input_path, f"{caseid}.txt"), 'r',encoding="utf8") as f:
                    query = process(f.read())
                scores= bm_25obj.get_scores(query)
                #print(len(scores))
                result[caseid]={}
                for score,file in zip(scores,li):
                    result[caseid][file]=score
            #result[caseid]=sorted(result[caseid].items(), key=lambda item: item[1],reverse=True)
        with open(os.path.join(output_path,f'scores_batch_{i}.json'),'w') as f:
            json.dump(result,f,indent=4)"""
            
        



for i in range(2,5):

    run_one_fold(i)
