import sklearn
import cv2
import os
import numpy as np

X = []
Y = []
#for file in os.listdir('data'):
#    if file.endswith('.jpg'):
#        img = cv2.imread('data/' + file)
#        print(list(img))
#        X.append(list(img))
img = cv2.imread('data/frame000040.jpg')
print(np.shape(img))
#X.append(list(img))
        
#print(X)
