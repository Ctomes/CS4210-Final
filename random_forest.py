# run from data directory
##########################################
# goal: predict position_vehicle from unprocessed images
##########################################


from read_class import readClass
import os
import numpy 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
import cv2


# compares s and t as unordered lists
def compare(s, t):
    return Counter(s) == Counter(t)

# test reading
print (readClass('frame029045.txt')[0]['position_vehicle'])

# returns each 'position-vehicle' entry from the file
def yFill(fileName) :
    
    classes = readClass(fileName)
    
    positions = []
    
    for vehicle in classes :
        positions.append(readClass('frame029045.txt')[0]['position_vehicle'])
    
    return positions

#gets all filenames w/o extension from dir
def filesIn() :
    
    X = []
    y = []
    
    for file in os.listdir('.'):
        if file.endswith(".txt"):
            y.append(yFill(file))
        elif file.endswith(".jpg"):
            X.append(cv2.imread(file))
            
    return X, y
                



X, y = filesIn()

print(X[0])
print(y[0])

# Initialize X train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# create and fit the tree
clf=RandomForestClassifier(n_estimators=20)
clf.fit(X_train,y_train)


    