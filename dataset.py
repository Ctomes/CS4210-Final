import os
#from PIL import Image
import importlib
import random

from image_zoom import crop_img
from read_class import readClass as read

import numpy as np
import cv2


class plateDataset():
    def __init__(self, folder, randomize=True, do_augment=False, ext='jpg', outshape=(1080, 1920)):
        #TODO: implement randomized access mapping for fNames to increase epoch independency
        self.ext = ext
        self.folder = folder
        self.fNames = []
        self.randorder = randomize
        self.do_augment = do_augment
        self.outshape = outshape

        #get all filenames w/o extension from dir
        for file in os.listdir(folder):
            if file.endswith(".txt"):
                self.fNames.append(file.split('.')[0])

        #get randomization map for dataset
        self.randmap = [n for n in range(0, len(self.fNames))]
        random.shuffle(self.randmap)
    
    def __len__(self):
        return len(self.fNames)

    def __getitem__(self, idx):
        #if we want to randomize, use randmap to get items in random order
        index = idx
        if self.randorder:
            index = self.randmap[idx]

        #get filenames
        imPath = f"{self.folder}/{self.fNames[index]}.{self.ext}"
        gtPath = f"{self.folder}/{self.fNames[index]}.txt"

        #get data
        im = cv2.imread(imPath)
        #im = Image.open(imPath)
        #im = np.asarray(im)
        #print("thing1", np.shape(im))
        #print("thing2", np.shape(cv2.imread(imPath)))
        gtInfo = read(gtPath)
        
        #generate mask
        gt = np.zeros(self.outshape)
        for item in gtInfo:
            x = item['position_plate']['x']
            y = item['position_plate']['y']
            x2 = x + item['position_plate']['width']
            y2 = y + item['position_plate']['height']

            gt[y:y2, x:x2] = 1
        #add random augmentation to input image
        if self.do_augment:
            im, gt = self.augment(im, gt)

        print(im.max())
        im = (np.transpose(im, (2, 0, 1))/im.max())
        return (im, (1-gt))

    def augment(self, im, gt):
        #TODO: add image augmentations to dataset, orientation changes, color shifts
        decision = [True, False]
        image = np.asarray(im)
        truth = gt
        
        #brightness
        image = self.brightness(image, 0.5, 1.5)

        return (image, truth)

    
    def brightness(self, img, low, high):
        value = random.uniform(low, high)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv = np.array(hsv, dtype = np.float64)
        hsv[:,:,1] = hsv[:,:,1]*value
        hsv[:,:,1][hsv[:,:,1]>255]  = 255
        hsv[:,:,2] = hsv[:,:,2]*value 
        hsv[:,:,2][hsv[:,:,2]>255]  = 255
        hsv = np.array(hsv, dtype = np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return img

    def neworder(self):
        #create new randomized mapping for access order
        self.randmap = random.shuffle(self.randmap)

#each individual picture is an item, but each item may return 1-many license plates
class LetterSegDataset():
    def __init__(self, folder, randomize=True, do_augment=False, ext='jpg', inshape = (1080, 1920), outshape=(144, 48)):
        #TODO: implement randomized access mapping for fNames to increase epoch independency
        self.ext = ext
        self.folder = folder
        self.fNames = []
        self.randorder = randomize
        self.do_augment = do_augment
        self.inshape = inshape
        self.outshape = outshape

        #get all filenames w/o extension from dir
        for file in os.listdir(folder):
            if file.endswith(".txt"):
                self.fNames.append(file.split('.')[0])

        #get randomization map for dataset
        self.randmap = [n for n in range(0, len(self.fNames))]
        random.shuffle(self.randmap)

    def __len__(self):
        return len(self.fNames)

    def __getitem__(self, idx):
        #if we want to randomize, use randmap to get items in random order
        index = idx
        if self.randorder:
            index = self.randmap[idx]

        #get filenames
        imPath = f"{self.folder}/{self.fNames[index]}.{self.ext}"
        gtPath = f"{self.folder}/{self.fNames[index]}.txt"

        #get data
        im = cv2.imread(imPath)
        #im = Image.open(imPath)
        #im = np.asarray(im)
        #print("thing1", np.shape(im))
        #print("thing2", np.shape(cv2.imread(imPath)))
        gtInfo = read(gtPath)
        
        #generate mask
        gt = np.zeros(self.inshape)
        for item in gtInfo:
            for key in item:
                if 'char' in key:
                    x = item[key]['x']
                    y = item[key]['y']
                    x2 = x + item[key]['width']
                    y2 = y + item[key]['height']

                    gt[y:y2, x:x2] = 1

        crops = []
        masks = []
        for item in gtInfo:
            x = item['position_plate']['x']
            y = item['position_plate']['y']
            w = item['position_plate']['width']
            h = item['position_plate']['height']

            crops.append(cv2.resize(crop_img(imPath, x, y, w, h), self.outshape))
            masks.append(cv2.resize(gt[y:y+h, x:x+w], self.outshape))
            

        #TODO: add random augmentation to input image
        #if self.do_augment:
        #    im, gt = self.augment(im, gt)

        #print(im.max())
        crops = [(np.transpose(crop, (2, 0, 1))/im.max()) for crop in crops] #permute/normalize the crop arrays
        masks = [1 - mask for mask in masks] #invert the masks

        return (crops, masks)