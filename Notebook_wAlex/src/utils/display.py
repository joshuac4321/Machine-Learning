#Returns the largest output of A2
from emnist_models.emnist_model import decode
import random 
import numpy as np

def get_predictions(a2):
    return(np.argmax(a2, 0))

#for x, y in matrix Y and matrix p (predictions), if x == y, true (true = 1, false = 0), 
#sum all true, divide by label
def get_accuracy(Y, p):
    return(np.sum(Y == p)/ Y.size)
    
#predictions are returned from the get_predictions function which accesses the argmax of A2 from the 
#first axis, returning the index
#find current image, find prediction and label, display image with .gray, .imshow
index = random.randint(1,40000)
def display_predictions(X, Y, a2):
    print("Accuracy:", get_accuracy(Y, get_predictions(a2))*100, "%")
    print("Prediction: ", decode(get_predictions(a2[:,index])))
    print("Label:", decode(Y[index]))
    print()

def display_results(X, Y, a2):
    index = input("Index: ")
    index = int(index)
    display_predictions(X, Y, a2)
    print()
    X = X*255
    img_array = X[:, index].reshape(28,28)
    img_array = img_array.T
    plt.imshow(img_array, cmap='gray')
    plt.show()