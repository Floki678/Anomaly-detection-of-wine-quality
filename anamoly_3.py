import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
csv1=pd.read_csv("winequality-white.csv")
csv2=pd.read_csv("winequality-red.csv")
csv1.drop("quality",axis=1,inplace=True)
csv2.drop("quality",axis=1,inplace=True)
#Km=KMeans(n_clusters=1, init=’k-means++’, n_init=10, max_iter=300, tol=0.0001, precompute_distances=’auto’, verbose=0, random_state=None, copy_x=True, n_jobs=1, algorithm=’auto’)
csv1=np.array(csv1)
csv2=np.array(csv2)
#print(csv1.shape[0])
c1=(np.sum(csv1,axis=0)/csv1.shape[0])#.reshape(11,1)
c2=(np.sum(csv2,axis=0)/csv2.shape[0])#.reshape(11,1)
#print(np.sum(np.power(c1-csv1,2),axis=1).shape)
#print(c2)
dis1=np.sum(np.power(np.sum(np.power(c1-csv1,2),axis=1),1/2))/csv1.shape[0]
dis2=np.sum(np.power(np.sum(np.power(c2-csv2,2),axis=1),1/2))/csv2.shape[0]
#print(2*dis1)
#print(2*dis2)
count=0
s=list()
ss=list()
cu=0
for i in csv1:
    #print(i)
    if(np.power(np.sum(np.power(c1-i,2)),1/2)<=3.5*dis1):
        cu=cu+1
        continue
    else:
        s.append(cu)
        cu=cu+1
count=0
cu=0
for i in csv2:
    #print(i)
    if(np.power(np.sum(np.power(c2-i,2)),1/2)<=3.5*dis2):
        cu=cu+1
        continue
    else:
        ss.append(cu)
        cu=cu+1

print("Red wine anomolies: ")        
print(ss)

print("\nWhite wine anomolies: ")        
print(s)
