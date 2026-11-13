from contourfind import grayscale, shape, contour
import pandas
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt


pilimg = Image.open(r"Notebook_wAlex\handwriting.png")

pilimg = pilimg.convert("L")

print(type(pilimg))
pilimg = np.array(pilimg)
        
picture = contour(pilimg)
plt.imshow(picture, cmap='gray')
plt.show()