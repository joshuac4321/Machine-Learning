#import neccesary libraries
import numpy as np
import pandas
from matplotlib import pyplot as plt
import sys
import random

np.set_printoptions(threshold=sys.maxsize)

#load data using pandas
pandadata = pandas.read_csv(r"c:\Users\Administrator\Downloads\emnist-balanced-train.csv\emnist-balanced-train.csv")

#convert data to np array for matrix operations
data = np.array(pandadata)

np.random.shuffle(data)
data = data.T

#shape of data
exm, imgsn = data.shape

# shuffle before splitting into dev and training sets

#validation set
yvalid = data[0]
xvalid = data[1:785]

#normalize
xvalid = xvalid/255.

#rotates image by 90 or 270 degrees, returns flattened image
def rotate(img):
    degree = random.randint(1,3)
    img = np.rot90(img.reshape(28,28), k=degree-2).flatten()
    return img

#adds noise to image by randomly selecting 20 pixels and changing them to 100, returns flattened image
def noise(img):

    #generate boolean array
    mask = np.random.choice(img.size, size = 20, replace = False)

    img[mask] = 100
    return img

#applies random augmentation to image, either rotates, adds noise, or both, returns flattened image
def augment(img):
    aug = random.randint(1,6)
    if aug == 1:
        img = rotate(img)
        
    if aug == 2:
        img = noise(img)

    if aug == 3:
        img = rotate(img)
        img = noise(img)
    return img

#initialize random params between -0.5 and 0.5, prevents vanishing or exploding gradients
def xavinit_params():
    xavinit = np.sqrt(6/(784+47))
    w1 = np.random.uniform(-xavinit, xavinit, (256, 784))
    b1 = np.random.rand(256, 1)-0.5
    w2 = np.random.uniform(-xavinit, xavinit, (47, 256))   
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
def one_hot(Y):
    one_hot_Y = np.zeros((47, Y.size))
    one_hot_Y[Y, np.arange(Y.size)] = 1
    return one_hot_Y

#backprop through the network
def back_prop(w2, z1, a1, z2, a2, N, X, Y):
    one_hot_Y = one_hot(Y)
    dz2 = a2 - one_hot_Y
    dw2 = dz2.dot(a1.T)
    db2 = dz2
    dz1 = w2.T.dot(dz2) * ReLU_deriv(z1)
    dw1 = dz1.dot(X.T)
    db1 = dz1
    return dw2, db2, dw1, db1

#subtracts matrix from matrix, makes the correct ones closer to 1, W := W − α ⋅ gradient
#update parameters with the gradient, W := W − α ⋅ gradient
def update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    return w1, b1, w2, b2