import pslegal as psl
import nltk
import NNP_extractor as npe
import os

path="./Resource/adv-hireinsharma-cases-docs/"

entries = os.listdir(path)

legal_doc=[]
#file_content=[]

for entry in entries:
    file_content=open(path+entry).read()
    legal_doc.append(nltk.word_tokenize(file_content))

test_doc = open('./105895179').read()

NNP_list = npe.start(test_doc)
psvectorizer = psl.PSlegalVectorizer()
psvectorizer.fit_legal(legal_doc)
psvectorizer.fit_doc(NNP_list)

comb=[]
for key in NNP_list:
    score=psvectorizer.get_score([key])
    comb.append((key,score))

new_list= sorted(comb, key = lambda x: x[1])
print(*new_list, sep = "\n")