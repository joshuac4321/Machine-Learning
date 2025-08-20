# import neccesary libraries
import numpy as np
import pandas
from matplotlib import pyplot as plt
import random
import math

#load data using pandas
pandadata = pandas.read_csv(r"C:\Users\chenl\Downloads\train.csv\train.csv")

#convert data to np array for matrix operations
data = np.array(pandadata).T

fig, axes = plt.subplots(2, 1)

idx = random.randint(1,30000)
#validation set
xvalid = data[1:785, idx]
xvalidNorm = np.where(xvalid > 0, 255, 0)
xvalidNorm = xvalidNorm.reshape(28, 28)

templist = []
templist2 = []
xcontourlist = []
ycontourlist = []

xcontour = np.zeros((784,1))
ycontour = np.zeros((784,1))

def mod(number, divisor):
    l = number//divisor
    return l*divisor-number

shapelist = []

def shape(imgarray, idx):
    img_array[idx]
    shape(idx)

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
img_array = img_array.reshape(784)
box = []
xbox = []
ybox = []
for x in range(len(img_array)):
    if img_array[x] != 0:
        img_array[x] = 255
        box.append(x)

for x in box:
    xbox.append(x//28)
    ybox.append(mod(x, 28))

xmax = xbox.index(max(xbox))
xmin = xbox.index(min(xbox))
ymax = ybox.index(max(ybox))
ymin = ybox.index(min(ybox))

yavg = (xmax+xmin)//2
xavg = ((2*ymin + mod(ymax-ymin,28))//2)

img_array[box[xavg]] = 500
img_array[box[yavg]] = 500

img_array = img_array.reshape(28, 28)

img_array[mod(box[xavg], 28)][box[yavg]//28] = 500

axes[0].imshow(xcontour_array, cmap='gray')
axes[1].imshow(ycontour_array, cmap='gray')
plt.imshow(img_array, cmap='gray')
plt.show()