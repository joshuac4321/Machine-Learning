from contourfind import grayscale, shape, contour
import pandas
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt


pandadata = Image.open(r"C:\Users\chenl\OneDrive\Documents\GitHub\Machine-Learning\Notebook_wAlex\handwriting.png")

data = np.array(pandadata)

m, n, l = data.shape

picture = np.zeros((m, n))

for x in range(m):
    for y in range(n):
        picture[x][y] = sum(data[x][y][0:2])/3
        
picture = contour(picture)
plt.imshow(picture, cmap='gray')
plt.show()