# Anomaly-detection-of-wine-quality

The main aim of this project is to detect anomalous wines to recognize the bad quality wines and hence to improve the wine quality. 
We will be using three different anomaly detection algorithms, Gaussian distribution, SVM and circular K-Means to detect the anomalous wines and a comparative study will be done for all the techniques used.
Anomalous points here can be defined as a point that remains unclustered. A point may remain un clustered due to larger Euclidian distance from the clusters centroids, or maybe due to a probability out of the distribution defined by us, or maybe due to the fact that it lies out of the decision boundaries defined by SVM.
The anomalous points may not only depend upon the similarity criteria but also upon the data clustering technique used. We try to make a comparative study between the three algorithms used here.

# Project Description

We used the Wine Quality Dataset from Kaggle to train our models. 
Here are the steps we will undertake:
1.	Perform random mixing of all the data points.  
2.	Normalize the dataset.
3.	Prepare the 3 different models for anomaly detection.
4.	Test the models and compare them to find the most accurate and reliable one.


## Architecture


### 1. Gaussian Distribution
Given the data set we have to estimate the Gaussian distribution for each of the features. For each feature we need to find parameters that fit the data. The Gaussian distribution is given by-
 p(x; µ, σ^2 ) = 1 /√ 2πσ^2* e ^− (x−µ)^2/ 2σ^2 ,
 where µ is the mean and σ 2 controls the variance.





### 2. SVM
Support vector machines (SVM) are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. Given a dataset, each marked as belonging to one or the other of two categories, an SVM training algorithm builds a model that assigns new examples to one category or the other. New examples are then mapped into that same space and predicted to belong to a category based on which side of the gap they fall.




### 3.Circular K-Means
K-means clustering is used for cluster analysis in data mining. K-means clustering aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean. We have used a modified version of this algorithm, where we try to find the points with distance greater than alpha  times the average distance of all the points from centroid.  



## Parameter Settings

In the circular K-means used here, we define a radius parameter, that is used to increase the average radius of the cluster by a factor of alpha, where alpha is the radius parameter. This is done for both the clusters i.e. red and white wine. We used a fixed value of radius parameter here, but some advanced tuning algorithm can be used to predict the individual value of all the clusters available, hence independent and different values of radius parameters for different clusters is possible.
In the Gaussian probability model, we estimate the mean(mu) and sigma of the dataset and we use them to construct our probability distribution model.
For the SVM model, we choose the nu and gamma parameters which would work best for our anomaly detection application.





## DATASET DESCRIPTION

Two datasets are included, related to red and white vinho verde wine samples, from the north of Portugal.
Our dataset consists of 4898 instances and 12 attributes.
The classes are ordered and not balanced (e.g. there are much more normal wines than excellent or poor ones). 

### Attribute Information:
1 - fixed acidity  
2 - volatile acidity  
3 - citric acid  
4 - residual sugar   
5 - chlorides  
6 - free sulfur dioxide   
7 - total sulfur dioxide  
8 - density  
9 - pH  
10 - sulphates   
11 - alcohol  

Output variable (based on sensory data):

12 - quality (score between 0 and 10)  


