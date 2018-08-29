import numpy as np 
from numpy import genfromtxt
from scipy.stats import multivariate_normal
from sklearn.metrics import f1_score

def read_dataset(filePath,delimiter=','):
    return genfromtxt(filePath, delimiter=delimiter)

def feature_normalize(dataset):
    mu = np.mean(dataset,axis=0)
    sigma = np.std(dataset,axis=0)
    return (dataset - mu)/sigma

def estimateGaussian(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.cov(dataset.T)
    return mu, sigma
    
def multivariateGaussian(dataset,mu,sigma):
    p = multivariate_normal(mean=mu, cov=sigma)
    return p.pdf(dataset)

def selectThresholdByCV(probs,gt):
    best_epsilon = 0
    best_f1 = 0
    f = 0
    stepsize = (max(probs) - min(probs)) / 1000;
    epsilons = np.arange(min(probs),max(probs),stepsize)
    for epsilon in np.nditer(epsilons):
        predictions = (probs < epsilon)
        f = f1_score(gt, predictions, average = "binary")
        if f > best_f1:
            best_f1 = f
            best_epsilon = epsilon
    return best_f1, best_epsilon

data = read_dataset('winequality-red.csv')
data = data[~np.isnan(data).any(axis=1)]

# data1 = read_dataset('winequality-white.csv')
# data1 = data1[~np.isnan(data1).any(axis=1)]

tr_data = data[:1300, :11]
cv_data = data[1300:, :]

# tr_data1 = data1[:4200, :11]
# cv_data1 = data1[4200:, :]

gt_data = np.zeros((cv_data.shape[0], 1), dtype = int)
for idx, quality in enumerate(cv_data[:, 11]):
    if quality > 3:
        gt_data[idx] = 0
    else:
        gt_data[idx] = 1

cv_data = np.delete(cv_data, 11, 1)

# gt_data1 = np.zeros((cv_data1.shape[0], 1), dtype = int)
# for idx, quality in enumerate(cv_data1[:, 11]):
#    if quality > 3:
 #       gt_data1[idx] = 0
  #  else:
   #     gt_data1[idx] = 1

# cv_data1 = np.delete(cv_data1, 11, 1)

mu, sigma = estimateGaussian(tr_data)
p = multivariateGaussian(tr_data,mu,sigma)

p_cv = multivariateGaussian(cv_data,mu,sigma)
fscore, ep = selectThresholdByCV(p_cv,gt_data)
outliers = np.asarray(np.where(p < ep))

# mu1, sigma1 = estimateGaussian(tr_data1)
# p1 = multivariateGaussian(tr_data1,mu1,sigma1)

# p_cv1 = multivariateGaussian(cv_data1,mu1,sigma1)
# fscore1, ep1 = selectThresholdByCV(p_cv1,gt_data1)
# outliers1 = np.asarray(np.where(p1 < ep1))

print("Wine anomolies: ")  
print(outliers)
# print(outliers1)
