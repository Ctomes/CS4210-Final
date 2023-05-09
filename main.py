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
from BoxStuff import imgcrops, sharpen_image
from read_class import readClass as read


def main():
  device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
  print(device)
  #Part 1: Image crop to License plate
  #model instantiation
  loadpath = 'ModelWeights/UNet173.pt'
  params = (3, 1920, 1080) #expected image size
  plateModel = UNet(params).to(device).eval()
  plateModel.load_state_dict(torch.load(loadpath, map_location=torch.device(device)))

  #image reading
  imgpath = input("Enter the filename for prediction: ")
  img = cv2.imread(imgpath)
  img = img/img.max() #normalization

  plateimgs = imgcrops(plateModel, img)
  
  print(len(plateimgs))
  for plateimg in plateimgs: 
    plt.imshow(cv2.resize(plateimg, (150, 50)))
    plt.show()
  

  #Part 2: License plate crop to letters
  #specify the model to use
  loadpath = 'ModelWeights/model2/UNetFinal.pt'
  params = (3, 144, 48) #this IS a hyperparameter kinda, whatever we decide is the standard
  letterModel = UNet(params).to(device).eval()
  letterModel.load_state_dict(torch.load(loadpath, map_location=torch.device(device)))

  # for each plate, resize to the predefined params
  resizedPlates = [cv2.resize(plate, params[1:]) for plate in plateimgs]

  # transform those plate images into sets of letter images
  letterSets = [imgcrops(letterModel, plate, erode = True, exp = 3) for plate in resizedPlates]

  for letterSet in letterSets:
     for letter in letterSet:
      plt.imshow(cv2.resize(letter, (28, 28)))
      plt.show()
     


  #Part 3: Individual letters from previous models to overall string prediction
  #may want to test with grayscaling the image first
  loadpath = 'Model3/UNet72.pt'
  params = (3, 28, 28) #this IS a hyperparameter
  letterSize = (28, 28)
  letterGuesser = convNet(params)
  letterGuesser.load_state_dict(torch.load(loadpath, map_location=torch.device(device))) 

  #translations for the model
  bank = []
  for i in range(26):
    bank.append(chr(i+65)) #use ascii offset
  for i in range(10):
    bank.append(i) 

  #predict all the plates
  plates = []
  for letterSet in letterSets:
    #normalize the size of the letters, then convert them to torch tensors for prediction
    resizedLetters = [cv2.resize(letter, letterSize) for letter in letterSet]
    resizedLetters = [sharpen_image(letter) for letter in resizedLetters] #sharpen the image
    
    #show the effect of sharpening
    for letter in resizedLetters:
      plt.imshow(letter)
      plt.show()
    
    resizedLetters = [torch.from_numpy(letter).float().permute(2, 0, 1).unsqueeze(0) for letter in resizedLetters]

    #get a letter prediction for each letter
    letters = [torch.argmax(letterGuesser(letter).cpu()) for letter in resizedLetters]
    letters = [bank[letter] for letter in letters]
    letters = [str(letter) for letter in letters]

    #join those letters into a single string
    plate = ''.join(letters)

    #append that string to the list of plates
    plates.append(plate)

  print(plates)

if __name__ == "__main__":
    main()