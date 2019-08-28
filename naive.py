import csv
import math
import operator
import numpy as np
import matplotlib.pyplot as plt
#Naive Bayes Classifier

#The data is first read into a main train array which will be selected later to change the training size
main_train_x = np.genfromtxt('./hw1data/propublicaTrain.csv', delimiter=',',skip_header=1, usecols=(1,2,3,4,5,6,7,8,9)) 
main_train_y = np.genfromtxt('./hw1data/propublicaTrain.csv', delimiter=',',skip_header=1, usecols=(0)) 
test_x = np.genfromtxt('./hw1data/propublicaTest.csv', delimiter=',',skip_header=1, usecols=(1,2,3,4,5,6,7,8,9)) 
test_y = np.genfromtxt('./hw1data/propublicaTest.csv', delimiter=',',skip_header=1, usecols=(0))

#the value of k is the size of the training dataset
k = 500
#naive[] saves a tuple with the accuracy and training size for plotting
naive = []
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
    #we have two arrays (one for each label) which will store the count values for each feature. Each feature count will be stores in a map
    #Thus, count_0 is an array of dictionaries where the index of a dictionary is the index of the feature being tracked.
    count_0  =  [{} for x in range(len(train_0[0]))]
    count_1 =  [{} for x in range(len(train_1[0]))]
    for i in range(len(train_0)):
        for j in range(len(train_0[i])):
            if train_0[i][j] not in count_0[j]:
                count_0[j][train_0[i][j]] = 1
            else:
                count_0[j][train_0[i][j]] += 1
    for i in range(len(train_1)):
        for j in range(len(train_1[i])):
            if train_1[i][j] not in count_1[j]:
                count_1[j][train_1[i][j]] = 1
            else:
                count_1[j][train_1[i][j]] += 1
    ## Begin Prediction
    # the following variables track the true positive, false positive, true negative and false negative values for both classes 
    #enoded as 0 =a, 1=b
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
    predictions=np.zeros(len(test_x))

    for i in range(len(test_x)):
        #the class priors are calculated first 
        pred_0 = len(train_0) / (len(train_0) + len(train_1))
        pred_1 = len(train_1) / (len(train_0) + len(train_1))
        #we iterate through each feature in the test case
        #smoothing is done here by incrementnig each count by 1, so if value is not present, count =0 so we increment by 1 to count=1
        for j in range(len(test_x[0])):
            #we now check whether the value of the feature is already present in the dictionary. If it is not, we add it
            if test_x[i][j] not in count_0[j].keys():
                pred_0 = pred_0*((1) / (sum(count_0[j].values()) + len(count_0[j])))
                count_0[j][test_x[i][j]] = 1
            #fi it is present, we get the value and calculate the probability
            else:
                pred_0 = pred_0*((count_0[j][test_x[i][j]] + 1)/ (sum(count_0[j].values())+len(count_0[j])))
            #we do the same for both labels.
            if test_x[i][j] not in count_1[j].keys():

                pred_1 = pred_1*((1) / (sum(count_1[j].values()) + len(count_1[j])))
                count_1[j][test_x[i][j]] = 1
            else:
                pred_1 = pred_1*((count_1[j][test_x[i][j]] + 1) / (sum(count_1[j].values()) +len(count_1[j])))
        #assign label
        if pred_0 > pred_1:
            predictions[i] = 0
        else:
            predictions[i] = 1
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
    k += 300
    if(len(train_x)!=len(main_train_x)):
        if k > len(main_train_x):  k=len(main_train_x)
    naive.append((k,(tn_a+tp_a+tn_b+tp_b)*100/len(test_y)))
    
dp_naive=np.asarray([(tp_a+fp_a)/tot_a,(tp_b+fp_b)/tot_b])
eo_naive=np.asarray(((tp_a)/tot_a,(tp_b)/tot_b,(tn_a)/tot_a,(tn_b)/tot_b ))
acc_naive=np.asarray(((tp_a+tn_a)/tot_a,(tp_b+tn_b)/tot_b))
pp_naive=np.asarray(((tp_a)/(tp_a+fp_a),(tp_b)/(tp_b+fp_b),(tn_a)/(tn_a+fn_a),(tn_b)/(tn_b+fn_b)))
print("Accuracy with entire training set", (tn_a+tn_b+tp_a+tp_b)*100/len(test_y))
zip(*naive)
plt.plot(*zip(*naive))
plt.xlabel('Training Size')
plt.ylabel('Accuracy')
plt.title('Naive Bayes')
plt.show()  