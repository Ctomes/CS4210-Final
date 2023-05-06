import random
from read_class import readClass as read
import numpy as np
import cv2
import os

class charDataset:
    def __init__(self, path):
        self.path = path
        self.data = []

        #for now image extension is just jpg, hard coded, could be updated for generalizability
        self.ext = '.jpg'

        
        #takes in dataset folder
        #pulls out txt files
        #for each character creates a tuple of a file identifier, position in image, what character it is, character position in string
        for fName in os.listdir(self.path):
            if fName.endswith('.txt'):
                fInfo = read(f"{self.path}/{fName}")
                for vehicle in fInfo:
                    i = 0
                    for key in vehicle.keys():
                        if key.startswith('char'):
                            self.data.append((fName.split('.')[0] + self.ext, vehicle[key], vehicle['plate'][i], i))
                            i += 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pos = self.data[idx][1]

        #extract image
        img = cv2.imread(f"{self.path}/{self.data[idx][0]}")

        #extract character
        img = img[pos['y']:(pos['y']+pos['height']), pos['x']:(pos['x']+pos['width']), :]

        #resize for uniform data shape 
        #NOTE: it might be a good idea to change the original characters os that their shape is square before resizing, but depends on how model 2 dataset works
        img = cv2.resize(img, (28, 28))
        return img, self.data[idx][2]

