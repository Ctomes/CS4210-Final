import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
#input, mask from model as a torch tensor
#output, array of boxes for mask from model
#boxes are quadruples of form: (x, y, h, w)
def getBoxes(imgpred, threshold=.9999, blur=True):
  #permute for using opencv stuff
  imgpred = imgpred.squeeze().unsqueeze(0) #removes the batch, and if grayscale, restores the channels
  prebox = imgpred.permute(1, 2, 0).detach().cpu().numpy()

  if blur:
    prebox = cv2.GaussianBlur(prebox, (5,5), cv2.BORDER_DEFAULT)
  prebox = cv2.threshold(prebox, threshold, 1, cv2.THRESH_BINARY)[1]

  plt.imshow(prebox)
  plt.show()

  #numpy only likes uint8 for finding contours
  prebox = prebox.astype(np.uint8)

  #find the contours using opencv
  contours, _ = cv2.findContours(prebox, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  #calculate the boxes from the contours
  boxes = []
  for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    boxes.append((x, y, w, h))

  return boxes

#used to crop an image, given the boxes to crop in form (x, y, w, h), and a numpy array of the image
#returns a list of cropped images (as numpy arrays)
def cropsFromBoxes(img, boxes):
  cropped_imgs = []
  for box in boxes:
    x, y, w, h = box
    cropped_imgs.append(img[y:y+h, x:x+w]) #are we sure this is the order?

  return cropped_imgs

#crops to a target, specified by a model
#model is the segmentation model to use
#img is a numpy array of an image
#returns a list of all the cropped targets
def imgcrops(model, img):
  #predict using the model
  torchimg = torch.from_numpy(img).float()
  torchimg = torchimg.permute(2, 0, 1)
  #print(torchimg.shape)
  imgpred = model(torchimg) #removes the batch and restores the channels, as well as remove gradient
  predflipped = 1 - imgpred

  #return an array of cropped images from the boxes generated from the model
  boxes = getBoxes(predflipped)
  cropped_imgs = cropsFromBoxes(img, boxes)

  return cropped_imgs