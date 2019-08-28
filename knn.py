import numpy as np
import csv
import math
import operator
import matplotlib.pyplot as plt

#KNN Classifier

#finds the distance between two arrays of same length 
def findDistance(a,b,len):
    return np.linalg.norm(a-b)

#finds the K nearest rows to the test case, and then implements voting to find the final label
def getNearest(test_case,k):
    distances=[]
    y_label=0

    for x in range(len(train_x)):
        dist=findDistance(test_case,train_x[x],len(test_case))
        distances.append((train_y[x],dist))
    #we find the closest distances for the test case
    distances.sort(key=operator.itemgetter(1))
    count_0=0
    count_1=0
    #we count the number of 0 and 1 labels and then decide on the final vote
    for i in range(k):
        if distances[i][0] ==0:
            count_0+=1
        else:
            count_1+=1
    if(count_1>count_0):
        return 1
    else:
        return 0
    
import csv
import math
import operator
import matplotlib.pyplot as plt

#The data is first read into a main train array which will be selected later to change the training size
main_train_x = np.genfromtxt('./hw1data/propublicaTrain.csv', delimiter=',',skip_header=1, usecols=(1,2,3,4,5,6,7,8,9)) 
main_train_y = np.genfromtxt('./hw1data/propublicaTrain.csv', delimiter=',',skip_header=1, usecols=(0)) 
test_x = np.genfromtxt('./hw1data/propublicaTest.csv', delimiter=',',skip_header=1, usecols=(1,2,3,4,5,6,7,8,9)) 
test_y = np.genfromtxt('./hw1data/propublicaTest.csv', delimiter=',',skip_header=1, usecols=(0))

#the value of k is the size of the training dataset
k = 500
#knn saves a tuple with the accuracy and training size for plotting
knn = []
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
    predictions=np.zeros(len(test_x))
    #we calculate the tp,tn, fn, fp values 
    for i in range(len(test_x)):
        predictions[i]=getNearest(test_x[i],15)
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
    knn.append((k,(tp_a + tp_b + tn_a + tn_b)*100/len(test_y)))

    if(len(train_x)!=len(main_train_x)):
        if k > len(main_train_x):  k=len(main_train_x)
dp_knn=np.asarray([(tp_a+fp_a)/tot_a,(tp_b+fp_b)/tot_b])
eo_knn=np.asarray(((tp_a)/tot_a,(tp_b)/tot_b,(tn_a)/tot_a,(tn_b)/tot_b ))
acc_knn=np.asarray(((tp_a+tn_a)/tot_a,(tp_b+tn_b)/tot_b))
pp_knn=np.asarray(((tp_a)/(tp_a+fp_a),(tp_b)/(tp_b+fp_b),(tn_a)/(tn_a+fn_a),(tn_b)/(tn_b+fn_b)))
print("Accuracy with entire training set and k=15:", (tn_a+tn_b+tp_a+tp_b)*100/len(test_y))
zip(*knn)
plt.plot(*zip(*knn))
plt.xlabel('Training Size')
plt.ylabel('Accuracy')
plt.title('KNN')
plt.show()   