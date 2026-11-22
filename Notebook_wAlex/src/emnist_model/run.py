import numpy as np
import pandas
import sys
from emnist_model import gradient_descent

np.set_printoptions(threshold=sys.maxsize)

#load data using pandas
pandadatatrain = pandas.read_csv(r"c:\Users\Administrator\Downloads\emnist-balanced-train.csv\emnist-balanced-train.csv")
pandadatavalid = pandas.read_csv(r"c:\Users\Administrator\Downloads\emnist-balanced-test.csv\emnist-balanced-test.csv")

#convert data to np array for matrix operations
datatrain = np.array(pandadatatrain)
datavalid = np.array(pandadatavalid)

np.random.shuffle(datatrain)
datatrain = datatrain.T

np.random.shuffle(datavalid)
datavalid = datavalid.T

#shape of data
m, n = datatrain.shape

m2, n2 = datavalid.shape

# shuffle before splitting into dev and training sets

#validation set
labelvalid = datavalid[0]
xvalid = datavalid[1:785]
xvalid = xvalid/255.

#training set, prevents overfitting
labeltrain = datatrain[0, :]
xtrain = datatrain[1:785, :]
xtrain = xtrain/255.

gradient_descent(xtrain, labeltrain, xvalid, labelvalid, 0.1, 10000, 1000)