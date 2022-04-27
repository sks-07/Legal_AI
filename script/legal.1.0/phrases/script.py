from re import T
import numpy as np
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier 
from sklearn import metrics 
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier
import json 
import os
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier,RadiusNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

def one_hot_vector(data1,case_no,advocate_list):
    vect=[0]*len(advocate_list)
    for adv in data1[case_no]:   
        vect[advocate_list.index(adv)]=1
    return vect

path="./Resource/20_fold/fold_8/high_count_advs_8.json"
path1="./Resource/20_fold/fold_8/case_targets_8.json"  
rep_path="./re/Result/"

f = open (path, "r")
data = json.loads(f.read())

f1 = open (path1, "r")
case_target = json.loads(f1.read())
adv_list=[]
for key in data:
    adv_list.append(key)

def load(path,advocate_list):
    l=os.listdir(path)
    train=[]
    target=[]
    i=0
    for k in l:
        train.append(np.load(path+k))
        vect=[0]*len(advocate_list)
        vect[advocate_list.index(k.split('.')[0])]=1
        target.append(vect)
        #print("\rTotal training cases loaded {}/{}".format(i+1,len(l)),end="")
        i+=1
    return train,target

def test_data(path,advocate_list):
    l=os.listdir(path)
    train=[]
    target=[]
    i=0
    for k in l:
        train.append(np.load(path+k))
        vect=one_hot_vector(case_target,k.split('.')[0],advocate_list)
        target.append(vect)
        #print("\rTotal Test cases loaded {}/{}".format(i+1,len(l)),end="")
        i+=1
    return train,target

def svm_classifier(path1,path2):
    x_train=[]
    y_train=[]
    x_test=[]
    y_test=[]
    i=1
    for key in data:
        for case_no in data[key]['train']:
            x_train.append(np.load(path1+case_no+".npy"))
            y_train.append(i)
        for case_no in data[key]['test']:
            x_test.append(np.load(path2+case_no+".npy"))
            y_test.append(i)
        i+=1

    model2 = svm.SVC(kernel='rbf', gamma=1, C=1, decision_function_shape='ovo') # Linear Kernel
    model2.fit(x_train, y_train)
    y_pred2 = model2.predict(x_test)
    print("\nAccuracy for svm:",metrics.accuracy_score(y_test, y_pred2))
    return x_train,y_train,x_test,y_test

def model_training(model,x_train,y_train,x_test,y_test,name):
    model.fit(x_train,y_train)
    ypred=model.predict(x_test)
    print(name)
    print("\t\t Accuracy :",metrics.accuracy_score(y_test, ypred))
    #print("\t\t Precision :",metrics.precision_score(y_test, ypred, average='macro'))
    print("\t\t Recall :",metrics.recall_score(y_test, ypred, average='macro'))

def neighplot(k,x_train,y_train,x_test,y_test):
    y=[]
    for i in range(1,k):
        model = KNeighborsClassifier(n_neighbors=i)
        model.fit(x_train,y_train)
        ypred=model.predict(x_test)
        y.append(metrics.accuracy_score(y_test, ypred))
    plt.plot(range(1,k),y)
    plt.xlabel("K value")
    plt.ylabel("Accuracy")
    plt.title("K vs Accuracy KNN sklearn")
    plt.show()
"""
X,Y=load(rep_path+"advocates/",adv_list)
print("\n")
X_test,Y_test=test_data(rep_path+"test_docs/",adv_list)

clf = DecisionTreeClassifier()
clf.fit(X,Y)

y_pred = clf.predict(X_test)
print("\nAccuracy for :",metrics.accuracy_score(Y_test, y_pred))

model = RandomForestClassifier()
model.fit(X,Y)
print("\n")
y_pred1 = model.predict(X_test)
print("\nAccuracy for :",metrics.accuracy_score(Y_test, y_pred1))

model2 = svm.SVC(kernel='rbf', gamma=1, C=1, decision_function_shape='ovo') # Linear Kernel
model2.fit(X, Y)
y_pred2 = model.predict(X_test)
print("\nAccuracy for :",metrics.accuracy_score(Y_test, y_pred2))
"""
x_train,y_train,x_test,y_test=svm_classifier("./re/Result/bm25advocates/train/","./re/Result/bm25advocates/test/")
clf = DecisionTreeClassifier()
model_training(clf,x_train,y_train,x_test,y_test,"DecisionTreeClassifier")
model = RandomForestClassifier()
model_training(model,x_train,y_train,x_test,y_test,"RandomForestClassifier")
extra_tree = ExtraTreeClassifier(random_state=0)
model_training(extra_tree,x_train,y_train,x_test,y_test,"ExtraTreeClassifier")
neigh = KNeighborsClassifier(n_neighbors=3)
model_training(neigh,x_train,y_train,x_test,y_test,"KNeighborsClassifier")
clf2 = GaussianNB()
model_training(clf2,x_train,y_train,x_test,y_test,"GaussianNB")
neighplot(15,x_train,y_train,x_test,y_test)



