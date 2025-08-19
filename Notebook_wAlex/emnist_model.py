#import neccesary libraries
import numpy as np
import pandas
from matplotlib import pyplot as plt
import sys
import random

np.set_printoptions(threshold=sys.maxsize)

#load data using pandas
pandadata = pandas.read_csv(r"C:\Users\chenl\Downloads\emnist-balanced-train.csv\emnist-balanced-train.csv")

#convert data to np array for matrix operations
data = np.array(pandadata).T

data = data.T
np.random.shuffle(data)
data = data.T

#shape of data
m, n = data.shape

# shuffle before splitting into dev and training sets

#validation set
labelvalid = data[0, :2000]
xvalid = data[1:785, :2000]
xvalid = xvalid/255.

#training set, prevents overfitting
labeltrain = data[0, 2000:]
xtrain = data[1:785, 2000:]
xtrain = xtrain/255.

#initialize random params between -0.5 and 0.5, prevents vanishing or exploding gradients
def init_params():
    xavierinit = np.sqrt(6/(784+10))
    w1 = np.random.uniform(-xavierinit, xavierinit, (47, 784))
    b1 = np.random.rand(47, 1)-0.5
    w2 = np.random.uniform(-xavierinit, xavierinit, (47, 47))   
    b2 = np.random.rand(47, 1)-0.5
    return w1, b1, w2, b2

#ReLU returns number if greater than zero, else returns 0
def ReLU(z):
    return(np.maximum(z,0))
    
#softmax activation function
def softmax(z):
    return (np.exp(z)/sum(np.exp(z)))

#forward propogation with 784 x ReLU 10 x softmax 10
def forwardprop(w1, b1, w2, b2, X):
    z1 = np.dot(w1, X) + b1
    a1 = ReLU(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2

#returns derivative of ReLU, 1 if greater than 0, 0 if less (true and false)
def ReLU_deriv(Z):
    return Z > 0

#one hot converts each number into a matrix
#ex. [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#     [0, 1, ...            0, 0, 0]
#     [0, 0, 1, ...         0, 0, 0]
#     ...                         0]]
def one_hot(Y):
    one_hot_Y = np.zeros((Y.max()+1, Y.size))
    one_hot_Y[Y, np.arange(Y.size)] = 1
    return one_hot_Y

#backprop through the network and adjust weights accordingly
#cross entropy loss chain rule
def back_prop(w2, z1, a1, z2, a2, X, Y):
    one_hot_Y = one_hot(Y)
    dz2 = a2.T - one_hot_Y.T 
    dz2 = dz2.T
    dw2 = 1 / n * dz2.dot(a1.T)
    db2 = 1 / n * np.sum(dz2)
    dz1 = w2.T.dot(dz2) * ReLU_deriv(z1)
    dw1 = 1 / n * dz1.dot(X.T)
    db1 = 1 / n * np.sum(dz1)
    return dw2, db2, dw1, db1, dz2, dz1

#subtracts matrix from matrix, makes the correct ones closer to 1, W := W− α ⋅ gradient
#update parameters with the gradient, W := W− α ⋅ gradient
def update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    return w1, b1, w2, b2

#Returns the largest output of A2
def get_predictions(a2):
    return(np.argmax(a2, 0))

#for x, y in matrix Y and matrix p (predictions), if x == y, true (true = 1, false = 0), 
#sum all true, divide by label
def get_accuracy(Y, p):
    # print(np.sum(Y == p))
    # print(Y.size)
    # print("psize")
    # print(p.size)
    return(np.sum(Y == p)/ Y.size)
    
#predictions are returned from the get_predictions function which accesses the argmax of A2 from the 
#first axis, returning the index
#find current image, find prediction and label, display image with .gray, .imshow
index = random.randint(1,40000)
def display_predictions(X, Y, a2):
    print("Accuracy:", get_accuracy(Y, get_predictions(a2))*100, "%")
    print("Prediction: ", decode(get_predictions(a2[:,index])))
    print("Label:", decode(Y[index]))
    print()

def display_predictions2(X, Y, a2):
    index = input("Index: ")
    index = int(index)
    print("Accuracy:", get_accuracy(Y, get_predictions(a2))*100, "%")
    print("Prediction: ", decode(get_predictions(a2[:,index])))
    print("Label:", decode(Y[index]))
    print()
    X = X*255
    img_array = X[:, index].reshape(28,28)
    img_array = img_array.T
    plt.imshow(img_array, cmap='gray')
    plt.show()

def decode(number):
    with open(r"C:\Users\chenl\OneDrive\Documents\GitHub\Machine-Learning\Notebook_wAlex\emnist_balanced_mapping.txt", 'r') as mapping:
        convert = mapping.readlines()
        for line in convert:
            linenum, unicodedecode = line.split()
            linenum = int(linenum)
            unicodedecode = chr(int(unicodedecode))
            if number == linenum:
                return unicodedecode

#apply forwardprop, backprop, and update params to run gradient descent, as well as for every 10th
#iteration, display image and 
def gradient_descent(X, Y, X2, Y2, alpha, iterations):
    iterations = iterations + 1
    w1, b1, w2, b2 = init_params()
    for x in range(iterations):
        z1, a1, z2, a2 = forwardprop(w1, b1, w2, b2, X)
        dw2, db2, dw1, db1, dz2, dz1 = back_prop(w2, z1, a1, z2, a2, X, Y)
        w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)
        
        if x % 10 == 0:
            print("Epoch:" + str(x))
            display_predictions2(X, Y, a2)
    while True:
        display_predictions2(X, Y, a2)
            # rand = random.randint(1,40000)
            # print(Y[rand:rand+37])
            # print(np.argmax(a2, 0)[rand:rand+37])
            # print(np.sum(Y[rand:rand+37] == (np.argmax(a2, 0)[rand:rand+37])))
            # print(a2.shape)
        # print("DW2")
        # print(dw2[0])
        # print("DB2")
        # print(db2)
        # print("DW1")
        # print(dw1[0])
        # print("DW1")
        # print(db1)
        # print("DZ2")
        # print(dz2)
        # print("DZ1")
        # print(dz1)
        # print("W1")
        # print(w1)
        # print("B1")
        # print(b1)
        # print("W2")
        # print(w2)
    """
        28
        34
        32
        """
gradient_descent(xtrain, labeltrain, xvalid, labelvalid, 0.1, 10000)