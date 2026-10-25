import random
import numpy as np
import pandas
from matplotlib import pyplot as plt

pandadata = pandas.read_csv(r"c:\Users\Administrator\Downloads\emnist-balanced-train.csv\emnist-balanced-train.csv")

data = np.array(pandadata)
data = data.T

img = data[1:785, 5]
def noise(img):
        imgcopy = img.copy()
        mask = np.random.choice(img.size, size = 20, replace = False)
        img[mask] = 100
        return img

test = np.zeros((40))
print(noise(img))
img = noise(img)
img = img.reshape(28,28)
plt.imshow(img, cmap='gray')
plt.show()