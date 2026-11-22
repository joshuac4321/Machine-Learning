import numpy as np

def ReLU(z):
    return(np.maximum(z,0))
    
#softmax activation function
def softmax(z):
    exp = np.exp(z)
    return exp / np.sum(exp, axis=0, keepdims=True)