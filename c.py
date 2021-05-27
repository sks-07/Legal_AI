import pslegal as psl
import nltk
import NNP_extractor as npe



path="./Resource/adv-hireinsharma-cases-docs/100588213"
path1 = "./Resource/adv-hireinsharma-cases-docs/4096809"
file_content = open(path).read()
tokens = nltk.word_tokenize(file_content)
file_content1 = open(path1).read()
tokens1 = nltk.word_tokenize(file_content1)


NNP_list = npe.start(file_content)
legal_tokenized_documents = [['law','reports','or','reporters','are','series','of','books','that','contain','judicial','opinions','from','a','selection','of','case','law','decided','by','courts'],
['when','a','particular','judicial','opinion','is','referenced,','the','law','report','series','in','which','the','opinion','is','printed','will','determine','the','case','citation','format'],
] #two legal documents

nonlegal_tokenized_documents = [['the','data','is','organized','into','20','different','newsgroups,','each','corresponding','to','a','different','topic'],
['some','of','the','newsgroups','are','very','closely','related','to','each','other'],
['the','following','file','contains','the','vocabulary','for','the','indexed','data'],
] #three non-legal documents


psvectorizer = psl.PSlegalVectorizer()
psvectorizer.fit(legal_tokenized_documents, nonlegal_tokenized_documents)
psvectorizer.fit_doc(NNP_list)
#print(psvectorizer.fit_doc(NNP_list))


#Then we use
phrase_score = psvectorizer.get_score(['allegation']) # if was trained using tokenized words
#print("\n",phrase_score)

print("\nNNP_list:",len(tokens1))
print("\nlegal token:",len(legal_tokenized_documents),len(legal_tokenized_documents[0]))