import pslegal as psl
import nltk
import NNP_extractor as npe
import os
from sklearn.datasets import fetch_20newsgroups
import json
from nltk.stem import PorterStemmer

path="./Resource/adv/"
#nl_path="./archive/"
dir_entries = os.listdir(path)
#entries1 = os.listdir(nl_path)
legal_doc=[]
nonlegal_doc=[]
legal_content=[]
file_entries=[]
nl_data = fetch_20newsgroups(remove = ('headers', 'footers', 'quotes')) 
ps = PorterStemmer()

for entries in dir_entries:
    files=os.listdir(path+entries)
    file_entries.append(files)
    for entry in files:
        file_content=open(path+entries+'/'+entry).read()
        legal_content.append(file_content)
        legal_doc.append(nltk.word_tokenize(file_content))

for i in range(len(nl_data.data)):
    nonlegal_doc.append(nltk.word_tokenize(nl_data.data[i])) 

psvectorizer = psl.PSlegalVectorizer()
psvectorizer.fit(legal_doc,nonlegal_doc)
l=len(file_entries)
save="./re/result/with_stemming/" 

file_length=[]
for fi in file_entries:
       file_length.append(len(fi))
s=0
file=[]
j=0
for l in file_length:
    print(s,s+l)
    files=legal_content[s:s+l]
    file.append(files)
    s+=l
    result={
    "Advocate_name":[],
    "case_no":[],
    "Phrase_and_get_score":[],
    "term_score":[],
             }
    print("\nCases for {} .....:".format(dir_entries[j]))
    for i in range(l):
        print("\n\tCase no {} :".format(file_entries[j][i]))
        NNP_list = npe.start(files[i])
        test_token=nltk.word_tokenize(files[i])
        print("\t\tTokenization complete.....")
        stemm=[]
        for w in test_token:
            #print(w, " : ", ps.stem(w))
            stemm.append(ps.stem(w))
        print("\t\tStemming complete.....")
        term_score=psvectorizer.fit_doc(stemm)
        print("\t\tDoc fitting complete.....")

        comb=[]
        #new_nnp_list=[]
        for key in NNP_list:
            ke=nltk.word_tokenize(key)
            ne=[]
            for k in ke:
                #print(k, " : ", ps.stem(k))
                ne.append(ps.stem(k))
            #new_nnp_list.append(ne)
            score=psvectorizer.get_score(ne)
            comb.append((key,score))

        new_list= sorted(comb, key = lambda x: x[1],reverse=True)
        result["Advocate_name"].append(dir_entries[j])
        result['case_no'].append(file_entries[j][i])
        result['Phrase_and_get_score'].append(new_list)
        result['term_score'].append(term_score)
        print("\tcases {}/{} completed.....".format(i+1,l))

    with open(save+dir_entries[j]+".json", "w") as write_file:
        json.dump(result, write_file)
    j=j+1   

