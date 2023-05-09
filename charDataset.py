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
        self.bank = dict()
        for i in range(65, 91):  # ASCII codes for uppercase A to Z
            self.bank[chr(i)] = i - 65
            #print(chr(i), ':', i - 65)
        
        for i in range(0, 10):
            self.bank[f'{i}'] = i + 26

        self.blank = np.zeros((36))

        
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
        #print(img.max(), img.min())

        #resize for uniform data shape 
        #NOTE: it might be a good idea to change the original characters os that their shape is square before resizing, but depends on how model 2 dataset works
        img = cv2.resize(img, (28, 28))
        img = self.sharpen_image(img)
        img = np.transpose(img, (2, 0, 1))

        #result = self.blank.copy()

        char = self.data[idx][2]
        result = self.bank[char]
        
        return img/255, int(result)

    def sharpen_image(self, image):
        # Create the sharpening kernel
        outside = 0.4
        inside = 5
        kernel = np.array([[-1,-1,-1], [-1,9.5,-1], [-1,-1,-1]])

        # Apply the kernel to the image using filter2D
        sharpened = cv2.filter2D(image, -1, kernel)

        return sharpened