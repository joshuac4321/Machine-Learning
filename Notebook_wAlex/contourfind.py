#import neccesary libraries
import numpy as np
import pandas
from matplotlib import pyplot as plt
import random

#load data using pandas
pandadata = pandas.read_csv(r"C:\Users\chenl\Downloads\train.csv\train.csv")

#convert data to np array for matrix operations
data = np.array(pandadata).T

#validation set
xvalid = data[1:785, 1]
xvalidNorm = np.where(xvalid > 0, 255, 0)
xvalidNorm = xvalidNorm.reshape(28, 28)

templist = []
templist2 = []
xcontourlist = []
ycontourlist = []

xcontour = np.zeros((784,1))
ycontour = np.zeros((784,1))

for x in range(len(xvalid)):
    if xvalid[x] > 0:
        templist.append(x)

    if xvalid[x] <= 0 and len(templist) != 0:
        xcontourlist.append(templist[0])
        xcontourlist.append(templist[-1])        
        templist = []

    xvalid = xvalid.reshape(28,28)
    
    xvalid = xvalid.T
    xvalid = xvalid.T
    xvalid = xvalid.T

    xvalid = xvalid.reshape(784)

    if xvalid[x] > 0:
        templist2.append(x)
    if xvalid[x] <= 0 and len(templist2) != 0:
        ycontourlist.append(templist2[0])
        ycontourlist.append(templist2[-1])        
        templist2 = []
    
    xvalid = xvalid.reshape(28,28)
    xvalid = xvalid.T
    xvalid = xvalid.reshape(784)

for x in xcontourlist:
    xcontour[x] = 255

for x in ycontourlist:
    ycontour[x] = 255

xcontour_array = xcontour.reshape(28,28)
ycontour_array = ycontour.reshape(28,28)
ycontour_array = ycontour_array.T

img_array = xcontour_array + ycontour_array
plt.imshow(img_array, cmap='gray')
plt.show()

