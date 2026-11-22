import numpy as np

"""one hot converts each number into a matrix"""
def one_hot(Y):
    one_hot_Y = np.zeros((Y.max()+1, Y.size))
    one_hot_Y[Y, np.arange(Y.size)] = 1
    return one_hot_Y
