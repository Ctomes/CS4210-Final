import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
#input, mask from model as a torch tensor
#output, array of boxes for mask from model
#boxes are quadruples of form: (x, y, h, w)
def getBoxes(imgpred, threshold=.9999, blur=True, boxexpand = 0):
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

  #sort boxes left to right
  boxes.sort()
  return boxes

#used to crop an image, given the boxes to crop in form (x, y, w, h), and a numpy array of the image
#returns a list of cropped images (as numpy arrays)
#cropexpand is because the model will predict a slightly smaller box than the actual box because of its training set
def cropsFromBoxes(img, boxes, cropexpand = 0):
  cropped_imgs = []
  for box in boxes:
    x, y, w, h = box

    x = x - cropexpand
    y = y - cropexpand
    w = w + cropexpand
    h = h + cropexpand

    cropped_imgs.append(img[y:y+h, x:x+w]) #are we sure this is the order?

  return cropped_imgs

#crops to a target, specified by a model
#model is the segmentation model to use
#img is a numpy array of an image
#returns a list of all the cropped targets
#erosion may help model 2 because the letters are close together, how much erosion is a hyperparameter
def imgcrops(model, img, thresh = .9999, blr = True, erode = False, exp = 0):
  if erode:
    img = cv2.erode(img, None, iterations = 1)
  #predict using the model
  torchimg = torch.from_numpy(img).float()
  torchimg = torchimg.permute(2, 0, 1)
  #print(torchimg.shape)
  imgpred = model(torchimg) #removes the batch and restores the channels, as well as remove gradient
  predflipped = 1 - imgpred

  plt.imshow(img)
  plt.show()
  if erode:
    img = cv2.dilate(img, None, iterations = 1)

  #return an array of cropped images from the boxes generated from the model
  boxes = getBoxes(predflipped, threshold=thresh, blur=blr)
  cropped_imgs = cropsFromBoxes(img, boxes, cropexpand = exp)

  return cropped_imgs

def sharpen_image(image):
    # Create the sharpening kernel
    outside = 0.4
    inside = 5
    kernel = np.array([[-1,-1,-1], [-1,9.5,-1], [-1,-1,-1]])

    # Apply the kernel to the image using filter2D
    sharpened = cv2.filter2D(image, -1, kernel)

    return sharpened