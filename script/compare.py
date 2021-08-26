import json

path1="./re/new_allcases/AtulKumar.json"
path2="./re/ALLCASES/AtulKumar.json"

with open(path1) as f: 
  data1 = json.load(f) 



print(data1['Phrase_and_get_score'])