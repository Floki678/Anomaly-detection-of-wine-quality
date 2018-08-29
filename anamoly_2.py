from sklearn import svm
import numpy as np
from numpy import genfromtxt

def read_dataset(filePath,delimiter=','):
    return genfromtxt(filePath, delimiter=delimiter)

data = read_dataset('winequality-red.csv')
data = data[~np.isnan(data).any(axis=1)]

data1 = read_dataset('winequality-white.csv')
data1 = data1[~np.isnan(data1).any(axis=1)]

tr_data = data[:1300, :11]
tr_data1 = data1[:4200, :11]

clf = svm.OneClassSVM(nu=0.05, kernel="rbf", gamma=0.1)
clf.fit(tr_data)
pred = clf.predict(tr_data)

clf1 = svm.OneClassSVM(nu=0.05, kernel="rbf", gamma=0.1)
clf1.fit(tr_data1)
pred1 = clf1.predict(tr_data1)

# inliers are labeled 1, outliers are labeled -1
normal = tr_data[pred == 1]
abnormal = tr_data[pred == -1]
            
normal1 = tr_data1[pred1 == 1]
abnormal1 = tr_data1[pred1 == -1]

outliers_red = []
outliers_white = []

print("No. of Red wine anomolies detected: ")  
print(abnormal.shape[0])
print("\nNo. of White wine anomolies detected: ")  
print(abnormal1.shape[0])

for row in abnormal:
    outliers_red.append(np.where(np.all(tr_data==row,axis=1)))

for row in abnormal1:
    outliers_white.append(np.where(np.all(tr_data1==row,axis=1)))

# print(outliers_red)
# print(outliers_white)
