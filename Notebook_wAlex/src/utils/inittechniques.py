import numpy as np
from math import sqrt

def xavinit(inputlen, classes, firstlayerlogits):
    xinit = sqrt(6/(inputlen+classes))
    w1 = np.random.uniform(-xinit, xinit, (firstlayerlogits, inputlen))
    b1 = np.zeros((firstlayerlogits, 1))
    w2 = np.random.uniform(-xinit, xinit, (classes, firstlayerlogits))   
    b2 = np.zeros((classes, 1))
    return w1, b1, w2, b2
