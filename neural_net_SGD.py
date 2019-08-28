import random
# Third-party libraries
import numpy as np
import scipy.io as spio 
import numpy as np
from random import seed
from random import randrange
import random
from csv import reader
from math import exp,tanh
import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))

class NeuralNet(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a
    
    def SGD(self, training_data, epochs, mini_batch_size, l_r):

        training_data = list(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches=[]
            for k in range(0, len(training_data), mini_batch_size):
                mini_batches.append([training_data[k:k+mini_batch_size]])

            for mini_batch in mini_batches:
                delta_b=[]
                delta_w=[]
                for b in self.biases:
                    delta_b.append(np.zeros(b.shape))
                for w in self.weights:
                    delta_w.append(np.zeros(w.shape))
                for x,y in mini_batch:
                    x = np.reshape(x, (-1,1))
                    y = np.reshape(y, (-1,1))
                    delta_t_b, delta_t_w = self.backprop(x, y)
                    for db, dtb in zip(delta_b, delta_t_b):
                        delta_b.append(db+dtb)
                    for dw, dtw in zip(delta_w, delta_t_w):
                        delta_w.append(db+dtb)


        ##### Gradient Descent without Adam 
                # self.weights = [w-(l_r/len(mini_batch))*nw
                # for w, nw in zip(self.weights, nabla_w)]
                # self.biases = [b-(l_r/len(mini_batch))*nb
                # for b, nb in zip(self.biases, nabla_b)]

        ##### Adam optimizer 

                beta_1 = 0.9
                beta_2 = 0.999
                epsilon = 1e-8
                
                               
                self.m_w  = []
                for m,nw in zip(self.m_w,delta_w):
                    self.m_w.append(np.multiply(beta_1,m) - np.multiply(1-beta_1, nw))


                self.v_w  = []
                for v,nw in zip(self.v_w,delta_w):
                    self.v_w.append(np.multiply(beta_2,v) - np.multiply(1-beta_2, nw*nw))
                self.m_w = np.asarray(self.m_w)
                self.v_w = np.asarray(self.v_w)
                v_w_temp =[]
                for v in self.v_w:
                    v_w_temp.append(np.sqrt(np.abs(v)))
                
                self.v_w = v_w_temp
                
                self.v_b  = []
                self.m_b  = []
                for m,nb in zip(self.m_b,delta_b):
                    self.m_b.append(np.multiply(beta_1,m) - np.multiply(1-beta_1, nb))
                    
                
                for v,nb in zip(self.v_b,delta_b):
                    self.v_b.append(np.multiply(beta_2,v) - np.multiply(1-beta_2, nb*nb))

                self.m_b = np.asarray(self.m_b)
                self.v_b = np.asarray(self.v_b)

                v_b_temp =[]
                for v in self.v_b:
                    v_b_temp.append(np.sqrt(np.abs(v)))

                self.v_b = v_b_temp

                
                for w,m,v in zip(self.weights,self.m_w,self.v_w):
                    self.weights = w- (l_r/len(mini_batch)) * (m/(v+epsilon))
                
                for b,m,v in zip(self.biases,self.m_b,self.v_b):
                    self.biases = b- (l_r/len(mini_batch)) * (m/(v +epsilon))
        
    def backprop(self, x, y):
        for b in self.biases:
            delta_b.append(np.zeros(b.shape))
        for w in self.weights:
            delta_w.append(np.zeros(w.shape))

        activation = x
        activations = [] 
        nets = [] 
        #feedforward We append the nets befpre activations as well since we need it when finding the deltas 
        for b, w in zip(self.biases, self.weights):
            activations.append(activation)
            net = np.dot(w, activation)+b
            activation = sigmoid(net)
            nets.append(net)
        
        delta = self.cost_function(activation, y) * sigmoid_der(net)
        #changes for output layer 
        delta_b[-1] = delta
        delta_w[-1] = np.dot(delta, activations[-2].T)

        for l in range(2, self.num_layers):
            net = nets[-l]
            der = sigmoid_der(net)
            delta = np.dot(self.weights[-l+1].T, delta) * der
            delta_b[-l] = delta
            delta_w[-l] = np.dot(delta, activations[-l-1].T)
        return (delta_b, delta_w)
    

    def cost_function(self, activation, y):
        return (activation-y)

def main():

    data = spio.loadmat('hw2_data.mat')
    x1 = data['X1']
    x2 = data['X2']
    y1 = data['Y1']
    y2 = data['Y2']
    dataset = zip(x2,y2/255)
    nn = NeuralNet((2,128,256,3))
    nn.SGD(dataset,2000,2048,0.001)

    pred = []
    for i in range(len(x2)):
        pred.append(nn.feedforward(np.reshape(x2[i],(-1,1))))
    plt.imshow(np.reshape(pred,(133,140,3)))
    plt.show()

if __name__ == '__main__':
    main()


