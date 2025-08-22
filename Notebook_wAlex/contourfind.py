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

def grayscale(img):
    for pixel in img:
        pixel = np.sum(pixel)/3

def mod(number, divisor):
    l = number//divisor
    return l*divisor-number

shapelist = [-29, -28, -27, -1, 1, 27, 28, 29]

def shape(img_array, idx):
    for index in shapelist:
        prev = idx
        if img_array(img_array[idx + index]) != 0:
            prev = idx
    prev = 0
    img_array[idx]     
    shape(idx)

def contour(img):
    m, n = img.shape

    img = img.reshape(m*n)

    templist = []
    templist2 = []
    xcontourlist = []
    ycontourlist = []

    xcontour = np.zeros((img.size,1))
    ycontour = np.zeros((img.size,1))
    for x in range(len(img)):
        img = img.reshape(m*n)
        if img[x] > 0:
            templist.append(x)

        if img[x] <= 0 and len(templist) != 0:
            xcontourlist.append(templist[0])
            xcontourlist.append(templist[-1])        
            templist = []

        img = img.reshape(m, n)
        
        img = img.T
        img = img.T
        img = img.T

        img = img.reshape(m*n)

        if img[x] > 0:
            templist2.append(x)
        if img[x] <= 0 and len(templist2) != 0:
            ycontourlist.append(templist2[0])
            ycontourlist.append(templist2[-1])        
            templist2 = []
        
        img = img.reshape(m, n)
        img = img.T

        for x in xcontourlist:
            xcontour[x] = 255

        for x in ycontourlist:
            ycontour[x] = 255

    xcontour_array = xcontour.reshape(m, n)
    ycontour_array = ycontour.reshape(m, n)
    ycontour_array = ycontour_array.T
    img_array = xcontour_array + ycontour_array
    return img_array

# img_array = contour(xvalid)
# img_array = img_array.reshape(784)
# box = []
# xbox = []
# ybox = []
# for x in range(len(img_array)):
#     if img_array[x] != 0:
#         img_array[x] = 255
#         box.append(x)

# for x in box:
#     xbox.append(x//28)
#     ybox.append(mod(x, 28))

# xmax = xbox.index(max(xbox))
# xmin = xbox.index(min(xbox))
# ymax = ybox.index(max(ybox))
# ymin = ybox.index(min(ybox))

# yavg = (xmax+xmin)//2
# xavg = ((2*ymin + mod(ymax-ymin,28))//2)

# img_array[box[xavg]] = 500
# img_array[box[yavg]] = 500

# img_array = img_array.reshape(28, 28)

# img_array[mod(box[xavg], 28)][box[yavg]//28] = 500

# plt.imshow(img_array, cmap='gray')
# plt.show()