import numpy as np

from utils.activations import ReLU, softmax
from utils.display import display_predictions, display_results
from utils.inittechniques import xavinit

class gradient_descent:
    def __init__(self, firstlayerlogits, batch_size, X, Y, alpha, iterations):
        self.firstlayerlogits = firstlayerlogits
        self.batch_size = batch_size
        self.X = X
        self.Y = Y
        self.alpha = alpha
        self.iterations = iterations

    #forward propogation with 784 x ReLU 10 x softmax 10
    def forwardprop(self, w1, b1, w2, b2, X):
        z1 = np.dot(w1, X) + b1
        a1 = ReLU(z1)
        z2 = np.dot(w2, a1) + b2
        a2 = softmax(z2)
        return z1, a1, z2, a2

    #backprop through the network and adjust weights accordingly
    #cross entropy loss chain rule
    def back_prop(self, w2, z1, a1, z2, a2, N, X, Y):
        one_hot_Y = one_hot(Y)
        dz2 = a2 - one_hot_Y
        dw2 = 1 / N * dz2.dot(a1.T)
        db2 = 1 / N * np.sum(dz2)
        dz1 = w2.T.dot(dz2) * z1 > 0
        dw1 = 1 / N * dz1.dot(X.T)
        db1 = 1 / N * np.sum(dz1)
        return dw2, db2, dw1, db1

    #subtracts matrix from matrix, makes the correct ones closer to 1, W := W − α ⋅ gradient
    #update parameters with the gradient, W := W − α ⋅ gradient
    def update_params(self, w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
        w1 = w1 - alpha * dw1
        b1 = b1 - alpha * db1
        w2 = w2 - alpha * dw2
        b2 = b2 - alpha * db2
        return w1, b1, w2, b2

    def decode(self, number):
        with open(r"data\emnist_balanced_mapping.txt", 'r') as mapping:
            convert = mapping.readlines()
            for line in convert:
                linenum, unicodedecode = line.split()
                linenum = int(linenum)
                unicodedecode = chr(int(unicodedecode))
                if number == linenum:
                    return unicodedecode

    #apply forwardprop, backprop, and update params to run gradient descent, as well as for every 10th
    #iteration, display image and 
    def gradient_descent(self):
        iterations = self.iterations + 1

        self.X.shape = (self.X.shape[0], self.X.shape[1])
        self.Y.shape = (self.Y.shape[0],)

        classes = len(np.unique(self.Y))
        inputlen = self.X.shape[0]

        w1, b1, w2, b2 = xavinit(inputlen, classes, self.firstlayerlogits)

        for x in range(iterations):
            z1, a1, z2,   a2 = self.forwardprop(w1, b1, w2, b2, self.X)
            dw2, db2, dw1, db1 = self.back_prop(w2, z1, a1, z2, a2, self.batch_size, self.X, self.Y)
            w1, b1, w2, b2 = self.update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, self.alpha)

            if x % 10 == 0:
                print("Epoch:" + str(x))
                self.display_predictions(self.X, self.Y, a2)
        
        with open("emnistparams.txt", "w") as file:
            file.write(str(w1))
            file.write(str(b1))
            file.write(str(w2))
            file.write(str(b2))
        
        while True:
            self.display_results(self.X, self.Y, a2)
