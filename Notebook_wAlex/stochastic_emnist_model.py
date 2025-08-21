#import neccesary libraries
import numpy as np
import pandas
from matplotlib import pyplot as plt
import sys
import random

np.set_printoptions(threshold=sys.maxsize)

#load data using pandas
pandadata = pandas.read_csv(r"C:\Users\chenl\Downloads\emnist-balanced-train.csv\emnist-balanced-train.csv")
pandadatavalid = pandas.read_csv(r"C:\Users\chenl\Downloads\emnist-balanced-test.csv\emnist-balanced-test.csv")

#convert data to np array for matrix operations
data = np.array(pandadata).T
datavalid = np.array(pandadatavalid).T

data = data.T
np.random.shuffle(data)

datavalid = datavalid.T
np.random.shuffle(datavalid)

#shape of data
m, n = data.shape

m2, n2 = datavalid.shape

# shuffle before splitting into dev and training sets

#validation set
labelvalid = data[:, 0]
xvalid = data[:, 1:785]
xvalid = xvalid/255.

#training set, prevents overfitting
xtrain = data[:, 0:785]

def augment(img):
    temp = img[0]
    img = img[1:785]
    aug = random.randint(1,6)
    if aug == 1:
        img = img.reshape(28,28)
        img = img.T
        img = img.reshape(784)
    if aug == 2:
        for pixel in img:
            if pixel != 0:
                pixel -= 100
    if aug == 3:
        for pixel in img:
            if pixel == 0:
                randpixel = random.randint(1,40)
                if randpixel == 1:
                    pixel = 100
    if aug == 4:
        img = img.reshape(28,28)
        img = img.T
        img = img.reshape(784)
        for pixel in img:
            if pixel >= 200:
                pixel -= 100
            if pixel == 0:
                randpixel = random.randint(1,40)
                if randpixel == 1:
                    pixel = 100
    
    img = np.insert(img, 0, temp) 

    return img

def init_params():
    xavierinit = np.sqrt(6/(784+47))
    w1 = np.random.uniform(-xavierinit, xavierinit, (47, 784))
    b1 = np.random.rand(47, 1)-0.5
    w2 = np.random.uniform(-xavierinit, xavierinit, (47, 47))   
    b2 = np.random.rand(47, 1)-0.5
    return w1, b1, w2, b2

def ReLU(z):
    return np.maximum(z, 0)

def softmax(z):
    return np.exp(z)/sum(np.exp(z))

def ReLU_deriv(n):
    return n > 0

def forwardprop(w1, b1, w2, b2, X):
    z1 = np.dot(w1, X) + b1
    a1 = ReLU(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = softmax(z2)
    print(w1.shape)
    print(X.shape)
    print(z1.shape)
    print(a1.shape)
    print(z2.shape)
    print(a2.shape)
    return z1, a1, z2, a2

def one_hot(Y):
    one_hot_Y = np.zeros(47)
    one_hot_Y[Y] = 1
    return one_hot_Y

def back_prop(w2, z1, a1, z2, a2, X, Y):
    one_hot_Y = one_hot(Y)
    dz2 = a2 - one_hot_Y
    dw2 = dz2.dot(a1)
    db2 = np.sum(dz2)
    dz1 = w2.dot(dz2) * ReLU_deriv(z1)
    # print(Y.shape)
    # print(X.shape)
    # print(a2.shape)
    # print(one_hot_Y.shape)
    # print(dz2.shape)
    # print(dw2.shape)
    # print(dz1.shape)
    dw1 = dz1.T.dot(X.T)
    db1 = np.sum(dz1)
    return dw2, db2, dw1, db1

(1,)
(784,)
(47, 47)
(47, 47)

(47, 112799)
(784, 112799)
(47, 112799)
(47, 47)

def update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    return w1, b1, w2, b2

def gradient_descent(X, X2, Y2, alpha, iterations, validiterations):
    iterations = iterations + 1
    w1, b1, w2, b2 = init_params()
    for y in range(iterations):
        for x in X:
            augmentX = augment(x)
            z1, a1, z2, a2 = forwardprop(w1, b1, w2, b2, augmentX[1:785])
            dw2, db2, dw1, db1 = back_prop(w2, z1, a1, z2, a2, augmentX[1:785], augmentX[0])
            w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)
        
            if x % 10 == 0:
                print("Epoch:" + str(x))
                # display_predictions(X, Y, a2)
    
    # for x in range(validiterations):
    #     z1, a1, z2, a2 = forwardprop(w1, b1, w2, b2, X2)
    #     dw2, db2, dw1, db1 = back_prop(w2, z1, a1, z2, a2, n2, X2, Y2)
    #     w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)

    #     if x % 10 == 0:
    #         print("Validation Epoch:" + str(x))
    #         display_predictions(X, Y, a2)

gradient_descent(xtrain, xvalid, labelvalid, 0.1, 100, 100)