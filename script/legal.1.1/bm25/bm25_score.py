
#import sys
#print(sys.version, sys.platform, sys.executable)

import json
import os
import re
import pickle



from string import punctuation

from tqdm import tqdm
from rank_bm25 import BM25Okapi
import nltk
#import numpy as np
#from gensim import corpora
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


def get_score(foldnum,batch_size=20):
    input_path=r"facts"
    bm=rf'rerun\setup2\bm25'
    output_path = os.path.join(bm,f"fold_{foldnum}")
    with open(
            rf"res\high_count_advs_{foldnum}.json",
            'r') as f:
        high_count_advs = json.load(f)

    adv_concat={}
    test_doc_ids = set()
    li=[]
    test_doc_id=[]

    for adv,case in high_count_advs.items():
        li=[caseid for caseid in case['db'] if(os.path.exists(os.path.join(input_path, f"{caseid}.txt")))]
        li=[caseid for caseid in case['train'] if(os.path.exists(os.path.join(input_path, f"{caseid}.txt")))]
        test_doc_ids.update(map(lambda x: x, case["test"]+case["val"]))

    test_doc_id=[k for k in test_doc_ids]
    with open(os.path.join(output_path,"bm25obj.sav"), "rb") as input_file:
        bm_25obj=pickle.load(input_file)

    if os.path.exists(os.path.join(output_path,f'remember.txt')):
        with open(os.path.join(output_path,f'remember.txt'),'r') as f:
            start_index=int(f.read())
            start_index+=batch_size
    else :
        start_index=0

    for i in tqdm(range(start_index,len(test_doc_id),batch_size)):
        result={}
        for caseid in tqdm(test_doc_id[i:i+batch_size]):
            if(os.path.exists(os.path.join(input_path, f"{caseid}.txt"))):
                with open(os.path.join(input_path, f"{caseid}.txt"), 'r',encoding="utf8") as f:
                    query = process(f.read())
                scores= bm_25obj.get_scores(query)
                #print(len(scores))
                result[caseid]=dict(zip(li,scores))
                #result[caseid]={k:v for k,v in sorted(dict(zip(scores,li)).items(),key=lambda item:item[1],reverse=True)}
            #result[caseid]=sorted(result[caseid].items(), key=lambda item: item[1],reverse=True)
        with open(os.path.join(output_path,f'remember.txt'),'w') as f:
            f.write(str(i))
        with open(os.path.join(output_path,f'scores_batch_{i}.json'),'w') as f:
            json.dump(result,f,indent=4)

for i in range(3,5):
    get_score(i)