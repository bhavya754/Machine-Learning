import numpy as np
import csv
import math
import operator
import matplotlib.pyplot as plt

#MLE Classifier

def get_gaussian(row):   
#Finds the mean and variance of the training data which is what the MLE on Gaussian Diistribution gives us                                                                                                                                                                            
    m = row.mean(axis=0) 
    var = np.matmul((row-m).transpose(), (row-m) )/ len(row)
    return m, var

def norm_pdf_multivariate(x, mu, sigma):
#Finds the value of the probability density function for a multivariate gaussian distribution
#Reference: https://stackoverflow.com/questions/11615664/multivariate-normal-density-in-python
    size = len(x)
    det=np.linalg.det(sigma)
    first = 1.0/ ( math.pow((2*math.pi),float(size)/2) * math.sqrt(det))
    x_mu = np.matrix(x - mu)
    inv = np.linalg.inv(sigma)        
    second = math.exp(-(x_mu * inv * np.transpose(x_mu))/2)
    return first * second

#The data is first read into a main train array which will be selected later to change the training size
main_train_x = np.genfromtxt('./hw1data/propublicaTrain.csv', delimiter=',',skip_header=1, usecols=(1,2,3,4,5,6,7,8,9)) 
main_train_y = np.genfromtxt('./hw1data/propublicaTrain.csv', delimiter=',',skip_header=1, usecols=(0)) 
test_x = np.genfromtxt('./hw1data/propublicaTest.csv', delimiter=',',skip_header=1, usecols=(1,2,3,4,5,6,7,8,9)) 
test_y = np.genfromtxt('./hw1data/propublicaTest.csv', delimiter=',',skip_header=1, usecols=(0))
#the value of k is the size of the training dataset
k = 500
#mle and x_ax store the accuracy of the classifier and size of the training set respectively
mle = []
x_ax=[]
while k <= len(main_train_x):
    train_x = main_train_x[:k]
    train_y = main_train_y[:k]
    train_0=[]
    train_1=[]
    #we split the dataset according to the label of the row
    for i in range(len(train_x)):
        if train_y[i]==0:
            train_0.append(train_x[i])
        else:
            train_1.append(train_x[i])
    train_0=np.asarray(train_0)
    train_1=np.asarray(train_1)
    #we get the mean and variance for both the gaussian distributions (for label=0 and label=1)
    mean_0,var_0=get_gaussian(train_0)
    mean_1,var_1=get_gaussian(train_1)
    #A small value is added to ensure that the matrix is not singular 
    var_0=var_0+0.0000001
    var_1=var_1+0.0000001
    predictions=np.zeros(len(test_x))
    # the following variables track the true positive, false positive, true negative and false negative values for both classes 
    #encoded as 0 =a, 1=b
    tp_a=0
    fp_a=0
    tn_a=0
    fn_a=0
    tp_b=0
    fp_b=0
    tn_b=0
    fn_b=0
    tot_a=0
    tot_b=0
    #the prior probabilities are calculated 
    prior_0=len(train_0)/len(train_x)
    prior_1=len(train_1)/len(train_x)
    for i in range(len(test_x)):
    	# for each test case, we find the pdf values for both gaussian distributions
        prob_0=norm_pdf_multivariate(test_x[i],mean_0,var_0) * prior_0
        prob_1=norm_pdf_multivariate(test_x[i],mean_1,var_1) * prior_1
        if(prob_0>prob_1):
            predictions[i]=0
        else:
            predictions[i]=1
        #we calculate the tp,tn, fn, fp values 
        if test_x[i][2]==0:
            if(predictions[i]==test_y[i]):
                if(test_y[i]==0):
                    tp_a+=1
                else:
                    tn_a+=1
            elif(predictions[i]==0 and test_y[i]==1):
                fp_a+=1
            else:
                fn_a+=1
            tot_a+=1
        else:
            tot_b+=1
            if(predictions[i]==test_y[i]):
                if(test_y[i]==0):
                    tp_b+=1
                else:
                    tn_b+=1
            elif(predictions[i]==0 and test_y[i]==1):
                fp_b+=1
            else:
                fn_b+=1
    #the values are saved for plotting
    dp_mle=np.asarray([(tp_a+fp_a)/tot_a,(tp_b+fp_b)/tot_b,(tn_a+fn_a)/tot_a,(tn_b+fn_b)/tot_b ])
    eo_mle=np.asarray(((tp_a)/tot_a,(tp_b)/tot_b,(tn_a)/tot_a,(tn_b)/tot_b ))
    acc_mle=np.asarray(((tp_a+tn_a)/tot_a,(tp_b+tn_b)/tot_b))
    pp_mle=np.asarray(((tp_a)/(tp_a+fp_a),(tp_b)/(tp_b+fp_b),(tn_a)/(tn_a+fn_a),(tn_b)/(tn_b+fn_b)))
    k += 300
    mle.append((tp_a + tp_b + tn_a + tn_b)*100/len(test_y))
    x_ax.append(k)
    if(len(train_x)!=len(main_train_x)):
    	if k > len(main_train_x):  k=len(main_train_x)
print("Accuracy with entire training set:", (tn_a+tn_b+tp_a+tp_b)*100/len(test_y))
#to plot the graph 
plt.plot(x_ax,mle)
plt.xlabel('Training Size')
plt.ylabel('Accuracy')
plt.title('MLE')
plt.show()