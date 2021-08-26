import json
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
save="./re/deadline/"
fi=[]
arr=[]
name=[]
#for num in range(2,5):
f = open (save+str(2)+"_gram"+"concatenated_scheme"+".json", "r")
data = json.loads(f.read())
fi.append(data)
"""i=0
ylab=[]
for key in data:
    a=[]
    for k in fi[0][key]:
        a.append(fi[0][key][k])
        if i==0:
            name.append(k)
    i+=1
    arr.append(a) 
    ylab.append(key)
    if i==223:
        break

fig, ax = plt.subplots(figsize=(500,500))
hm = sn.heatmap(data=arr,annot=False)
ax.set(xticks=np.arange(len(name)),yticks=np.arange(len(ylab)))
ax.set_xticklabels(name, rotation = 90,fontsize=5 )
ax.set_yticklabels(ylab, rotation = 45,fontsize=5 )
plt.show()"""
arr=[]
xlab=[]
ylab=[]
i=0
for key in data:
    a=[]
    b=[]
    x=fi[0][key]
    z=dict(sorted(x.items(), key=lambda item: item[1],reverse=True))
    for k in z:
        a.append(z[k])
        b.append(k)
    arr.append(a)
    xlab.append(b)
    ylab.append(key)

#fig, ax = plt.subplots(2, 2)
for i in range(3):
    plt.subplot(3, 3, i+1)
    plt.plot(np.arange(5),arr[i][:5])
    plt.xticks(np.arange(5), xlab[i][:5], rotation ='vertical')
    plt.title(ylab[i])
    
#fig.tight_layout(pad=3.0)
plt.show()   
