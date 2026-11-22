import random
import numpy as np

#rotates image by 90 or 270 degrees, returns flattened image
def rotate(img):
    degree = random.randint(1,3)
    img = np.rot90(img.reshape(28,28), k=degree-2).flatten()
    return img

#adds noise to image by randomly selecting 20 pixels and changing them to 100, returns flattened image
def noise(img):

    #generate boolean array
    mask = np.random.choice(img.size, size = 20, replace = False)

    img[mask] = 100
    return img

#applies random augmentation to image, either rotates, adds noise, or both, returns flattened image
def augment(img):
    aug = random.randint(1,6)
    if aug == 1:
        img = rotate(img)
        
    if aug == 2:
        img = noise(img)

    if aug == 3:
        img = rotate(img)
        img = noise(img)
    return img