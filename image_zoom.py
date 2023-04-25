# position_plate: 1822 436 81 27
# first num: x-coord of top left vertex of bounding box
# second nhum: y-coord of top left vertex of bounding box
# third num: width of bounding box
# fourth num: height of bounding box

import cv2

def crop_img(img_file, x, y, width, height):    
  # Load the input image
  img = cv2.imread(img_file)

  # Crop the image using the bounding box coordinates
  cropped_img = img[y:y+height, x:x+width]

  return cropped_img

  # Display for testing
  # cv2.imshow('Cropped Image', cropped_img)
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()