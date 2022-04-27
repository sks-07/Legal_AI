import pslegal as psl
import nltk
#import NNP_extractor as npe
from sklearn.datasets import fetch_20newsgroups
import json
#from nltk.stem import PorterStemmer
import pickle
import extract_noun_phrases as npe

path="./Resource/high_count_advs.json"
file_path="./Resource/masked_continuous_doc/"
#save="./re/phrases/noun_phrase_training/"
save="./re/phrases/tokenized_training/"

save_model="./re/saved_model/"
filename = 'tokenized_model.sav'
#filename = 'Allcase_model.sav'

f = open (path, "r")
data = json.loads(f.read())

psl = pickle.load(open(save_model+filename, 'rb'))
k=0
for key in data:
    result={
        key:{
            "Case_number":[],
            "Phrase_and_get_score":[],
        }       
    }
    print("\nCases of advocate {}".format(key))
    for entries in data[key]:#this loop will go through all train and test cases for advocates 
        for entry in data[key][entries]:
            _,case_no=entry.split('_')#extracting the case number 
            print("\n\tProcessing started for Case no:{} ".format(case_no))
            file_content=open(file_path+case_no+".txt").read()#reading file content 
            NNP_list = npe.extract(file_content)
            print("\t\tNoun extraction completed.....")
            psl.fit_doc(NNP_list)
            print("\t\tdoc fitting complete.....")
            comb=[]
            for nn in NNP_list:
                score=psl.get_score([nn])
                comb.append((nn,score))

            new_list= sorted(comb, key = lambda x: x[1],reverse=True)
            result[key]['Case_number'].append(case_no)
            result[key]['Phrase_and_get_score'].append(new_list)
            print("\tScoring completed....")
            k+=1
            print("\tTotal cases completed :{}".format(k))

    with open(save+key+".json", "w") as write_file:
        json.dump(result, write_file,indent = 4)

    
  