import json
import os
import numpy as np
"""
Setup1:
With advocate concatenation
"""
def map_score(doc,data1):
    mAP=0
    j=0
    for key in doc:
        #Ranking the advocates acoording to the non normailized bm25 score  
        sorted_dict=dict(sorted(doc[key].items(), key=lambda item: item[1],reverse=True)) 
        adv_list=[]#list to store the advocates names according to ranking
        for k in sorted_dict:
            adv_list.append(k)
            #print(k,sorted_dict[k])
        one_hot_vect=one_hot_vector(key,adv_list,data1)#creating one hot vector 
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
    



def one_hot_vector(case_no,advocate_list,data1):
    vect=[0]*len(advocate_list)#create a vector of length of total advocates
    for adv in data1[case_no]:   
        vect[advocate_list.index(adv)]=1#assigning 1 to index of associate advocate with a particular case number
    return vect
 
"""
With Setup:2
With ranking along all training cases
"""

def new_map(doc,case_to_adv):
    score=[]
    ap={}
    for testid,rlist in doc.items():
        #list of actual advocates associated with testid 
        test_adv=case_to_adv[testid]
        i=1
        r=0
        j=0
        check_adv=set()
        scores=[]
        for case in rlist:
            #if length of the list of original becomes zero then no need to check further
            if not len(test_adv):
                break

            #list of advocate associated with case in ranklist 
            maplist=case_to_adv[case]
            # match_adv=[]
            # flag to check relevance of the case
            flagm=0
            #flag to check the repeatedness of previous advocates
            flagr=0
            for adv in test_adv:
                if adv in check_adv:
                    flagr=1


            for adv in test_adv:
                if adv in maplist:
                    #check if actual adv  present in ranklist adv if yes then the document is relevant
                    flagm=1
                    check_adv.add(adv)
                    #removing the seen adv from the actual adv list
                    test_adv.remove(adv)
                    
            if flagm:
                #number of relevant document
                r+=1
                #appending AP in scores
                scores.append(r/i)          
            
            #skip the file if previously seen advocate occurs between next next advocate
            if flagr==0: 
                i+=1
            #print(r)
        if r:
            score.append(sum(scores)/r)
            ap[testid]=sum(scores)/r
        else:
            score.append(0)
            ap[testid]=0
        
    return np.mean(score),ap


"""
Setup 3:

Adding up score of relevant score
"""
def create_dict(path):
    with open(path,'r') as f:
        s=f.read()
    s=s.split('\n')[:-1]
    return dict(zip(s,len(s)*[0]))
    

def setup_3(original_score,case_target):
    new_score={}
    new_data={}
    for k,v in original_score.items():
        adv_map=create_dict('adv_list.txt')
        for adv,sc in v.items():
            for ad in case_target[adv]:
                adv_map[ad]+=sc
        new_score[k]=adv_map
    
    for k,v in new_score.items():
        new_data[k]={adv:scr for adv,scr in sorted(v.items(),key=lambda item:item[1],reverse=True)}

    # with open('eval\setup_3.json','w') as f:
    #     json.dump(new_data,f,indent=4)

    return map_score(new_data,case_target)

    

    


            
def check(doc,case_to_adv):
    r={}
    for testid,rlist in doc.items():
        r[testid]={}
        r[testid]['original']=case_to_adv[testid],
        for case in rlist:
            r[testid][case]=case_to_adv[case]
    with open('check_list.json','w') as f:
        json.dump(r,f,indent=4)


def main(foldnum):
    path=rf"eval\sim_scores_0.json"
    path1=rf"case_targets.json"  #case to advocate map file

    f = open (os.path.join(path), "r")
    data = json.loads(f.read())
    new_data={}
    for k,v in data.items():
        new_data[k]={adv:scr for adv,scr in sorted(v.items(),key=lambda item:item[1],reverse=True)}

    f1 = open (path1, "r")
    data1 = json.loads(f1.read())

    #print(data1['100416388'])
    #score,ap=new_map(new_data,data1)
    score=setup_3(new_data,data1)
    print("score: ",score)
    with open(rf'eval\map_setup_3_1.txt','w') as f:
        f.write(str(score))


    # with open(rf'eval\per_query_ap.json','w') as f:
    #     json.dump(ap,f,indent=4)
    #check(data,data1)
    #my_adv_list=[]
    #for key in data:
    #    my_adv_list.append(key)

    #adv_labels,test_labels=labels(data)


#for i in range(5):
#    main(i)
main(1)