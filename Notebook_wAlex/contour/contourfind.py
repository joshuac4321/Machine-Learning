# import neccesary libraries
import numpy as np
import pandas
from matplotlib import pyplot as plt
import random
import math

def grayscale(img):
    x, y, z = img.shape
    imgcopy = img
    print(img)
    for x in range():
        pixel = np.sum(pixel)/3
    print(img)
    if imgcopy.all() == img.all():
        print('true')
    return img

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

    xcontourlist = []
    ycontourlist = []

    xcontour = np.zeros((img.size,1))
    ycontour = np.zeros((img.size,1))
    
    img = img.reshape(m*n)

    threshold = 125
    prevpix = 0
    for idx, pix in enumerate(img):

        abspixdif = abs(pix - prevpix) 

        if abspixdif > threshold:
            templist.append(pix)

            if len(templist) != 0:
                for elem in templist:
                    xcountourlist.append(elem)
                templist = []

        prevpix = pix

    img = img.reshape(m, n)
    
    img = img.T

    img = img.flatten()

    
    prevpix = 0
    for idx, pix in enumerate(img):

        abspixdif = abs(pix - prevpix) 

        if abspixdif > threshold:
            templist.append(pix)

            if len(templist) != 0:
                for elem in templist:
                    ycountourlist.append(elem)
                templist = []

    for x in xcontourlist:
        xcontour[x] = 255

    for x in ycontourlist:
        ycontour[x] = 255

    xcontour_array = xcontour.reshape(m, n)
    ycontour_array = ycontour.T.reshape(m, n)
    img_array = xcontour_array + ycontour_array
    img_array = np.floor(img_array/2)
    return img_array
