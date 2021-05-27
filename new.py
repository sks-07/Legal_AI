import nltk
import pslegal as psl
import NNP_extractor as npe


path="./Resource/adv-hireinsharma-cases-docs/100588213"
file_content = open(path).read()
tokens = nltk.word_tokenize(file_content)
#print(tokens)

NNP_list = npe.start(file_content)
#print(NNP_list[10])
#print(type(tokens))

legal_tokenized_documents=tokens
psvectorizer = psl.PSlegalVectorizer()
psvectorizer.fit_legal(legal_tokenized_documents)
psvectorizer.fit_doc(tokens)
phrase_score = psvectorizer.get_score(NNP_list)
print("\n",phrase_score)