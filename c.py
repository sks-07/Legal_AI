import pslegal as psl
import nltk
import NNP_extractor as npe
import os
from sklearn.datasets import fetch_20newsgroups

path="./Resource/adv-hireinsharma-cases-docs/"
#nl_path="./archive/"
entries = os.listdir(path)
#entries1 = os.listdir(nl_path)
legal_doc=[]
nonlegal_doc=[]
legal_content=[]
nl_data = fetch_20newsgroups(remove = ('headers', 'footers', 'quotes'))

for entry in entries:
    file_content=open(path+entry).read()
    legal_content.append(file_content)
    legal_doc.append(nltk.word_tokenize(file_content))

#for entry in entries1:
#    file_content=open(nl_path+entry,'rb').read()
#    nonlegal_doc.append(nltk.word_tokenize(file_content))

for i in range(len(nl_data.data)):
    nonlegal_doc.append(nltk.word_tokenize(nl_data.data[i]))

"""
f=open("check.txt",'a')
for entry in entries:
    file_content=open(path+entry).read()
    f.write(file_content)
f.close()
"""


test_doc = open('./check.txt').read()
test_token=nltk.word_tokenize(test_doc)
#NNP_list=[]

#for doc in legal_content:
#    NNP_list.append(npe.start(doc))
NNP_list = npe.start(test_doc)
psvectorizer = psl.PSlegalVectorizer()
psvectorizer.fit(legal_doc,nonlegal_doc)
psvectorizer.fit_doc(test_token)

comb=[]
for key in NNP_list:
    score=psvectorizer.get_score([key])
    comb.append((key,score))

new_list= sorted(comb, key = lambda x: x[1])
f=open("results.txt",'w')
for element in new_list:
    f.write(element)

f.close()
print("completed")
#print(*new_list, sep = "\n")