#-------------------------------------------------------------------------
# AUTHOR: Christopher Tomes, Anthony Seward, Jason Rowley, ...
# FILENAME: main.py
# SPECIFICATION: This code will make a prediction using the full pipeline and the three pretrained models
# TIME SPENT: A few years
#-----------------------------------------------------------*/

#imports
import torch
import torch.nn as nn
import torchvision

import matplotlib.pyplot as plt

import cv2
import pandas as pd

from Model import UNet, convNet
from BoxStuff import imgcrops
from read_class import readClass as read

gtInfo = read("data/frame034160.txt")
print(gtInfo)

def main():
  #Part 1: Image crop to License plate
  #model instantiation
  loadpath = ''
  params = (3, 1920, 1080) #expected image size
  plateModel = UNet(params)
  plateModel.load_state_dict(torch.load(loadpath))

  #image reading
  imgpath = input("Enter the filename for prediction")
  img = cv2.imread(imgpath)

  plateimgs = imgcrops(plateModel, img)



  #Part 2: License plate crop to letters
  #specify the model to use
  loadpath = ''
  params = (3, 150, 50) #this IS a hyperparameter kinda, whatever we decide is the standard
  letterModel = UNet(params)
  letterModel.load_state_dict(torch.load(loadpath))

  # for each plate, resize to the predefined params
  resizedPlates = [cv2.resize(plate, params) for plate in plateimgs]

  # transform those plate images into sets of letter images
  letterSets = [imgcrops(letterModel, plate) for plate in resizedPlates]



  #Part 3: Individual letters from previous models to overall string prediction
  #may want to test with grayscaling the image first
  loadpath = ''
  params = (3, 28, 28) #this IS a hyperparameter
  letterGuesser = convNet(params)
  letterGuesser.load_state_dict(torch.load(loadpath))

  #predict all the plates
  plates = []
  for letterSet in letterSets:
    #normalize the size of the letters
    resizedLetters = [cv2.resize(letter, params) for letter in letterSet]

    #get a letter prediction for each letter
    letters = [letterGuesser(letter) for letter in resizedLetters] #this line is likely incorrect
    #most likely the letterguesser will return a number, not a string, something that will likely need to be smoothed out

    #join those letters into a single string
    plate = ''.join(letters)

    #append that string to the list of plates
    plates.append(plate)

  print(plates)