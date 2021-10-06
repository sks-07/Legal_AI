import json

def map_score(doc):
    mAP=0
    j=0
    for key in doc:
        #Ranking the advocates acoording to the non normailized bm25 score  
        sorted_dict=dict(sorted(doc[key].items(), key=lambda item: item[1],reverse=True)) 
        adv_list=[]#list to store the advocates names according to ranking
        for k in sorted_dict:
            adv_list.append(k)
            #print(k,sorted_dict[k])
        one_hot_vect=one_hot_vector(key,adv_list)#creating one hot vector 
        #print(one_hot_vect)
        sum=0
        p_k=0
        for i in range(len(adv_list)):
            if one_hot_vect[i]:#if entry in one hot vector is 1 then calculate the p@k and sum it calucate AP
                sum+=one_hot_vect[i]
                p_k+=sum/(i+1)
        mAP+=p_k/sum#adding AP 
        j+=1
    mAP=mAP/j#calculating mean of AP
    return mAP        
    



def one_hot_vector(case_no,advocate_list):
    vect=[0]*len(advocate_list)#create a vector of length of total advocates
    for adv in data1[case_no]:   
        vect[advocate_list.index(adv)]=1#assigning 1 to index of associate advocate with a particular case number
    return vect



path="./re/deadline/facts/phrase_concatenated_scheme.json"
path1="./Case_to_adv.json"  #case to advocate map file

f = open (path, "r")
data = json.loads(f.read())

f1 = open (path1, "r")
data1 = json.loads(f1.read())

#print(data1['100416388'])
print("score: ",map_score(data))

#my_adv_list=[]
#for key in data:
#    my_adv_list.append(key)

#adv_labels,test_labels=labels(data)
