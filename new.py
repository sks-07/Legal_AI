import nltk
#import pslegal as psl
import os
#import NNP_extractor as npe
#from top2vec import Top2Vec
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(400)
stemmer = SnowballStemmer("english")


path="./Resource/adv-hireinsharma-cases-docs/"
entries = os.listdir(path)
#entries1 = os.listdir(nl_path)
legal_doc=[]

legal_content=[]

for entry in entries:
    file_content=open(path+entry).read()
    legal_content.append(file_content)
    legal_doc.append(nltk.word_tokenize(file_content))


def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
# Tokenize and lemmatize
def preprocess(text):
    result=[]
    for token in gensim.utils.simple_preprocess(text) :
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
            
    return result

processed_docs = legal_doc
"""
processed_docs = []

for doc in legal_content:
    processed_docs.append(preprocess(doc))"""

dictionary = gensim.corpora.Dictionary(processed_docs)

bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

lda_model =  gensim.models.LdaMulticore(bow_corpus, 
                                   num_topics = 8, 
                                   id2word = dictionary,                                    
                                   passes = 10,
                                   workers = 2)

for idx, topic in lda_model.print_topics(-1):
    print("Topic: {} \nWords: {}".format(idx, topic ))
    print("\n")
"""
#tokens = nltk.word_tokenize(file_content)
#print(tokens)
model = Top2Vec(legal_content)
#NNP_list = npe.start(file_content)
#print(NNP_list[10])
#print(type(tokens))
print("Number of topics",model.get_num_topics())
topic_sizes, topic_nums = model.get_topic_sizes()
topic_words, word_scores, topic_nums = model.get_topics(model.get_num_topics())
print("\nTopic words : ",topic_words)
"""