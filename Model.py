import torch.nn as nn
import torch

### PRODUCTION MODELS ###
class UNet(nn.Module):
  def __init__(self, insize):
    super().__init__()

    #format of function names: FunctionType Layer#
    self.convin = nn.Conv2d(insize[0], 16, 8, padding = 'same')

    #downscaling
    #the input depth of every layer is the output depth of the last layer
    self.convolve1 = nn.Conv2d(16, 32, 3, padding = 'same')
    self.convolve2 = nn.Conv2d(32, 64, 4, padding = 'same')
    self.convolve3 = nn.Conv2d(64, 128, 3, padding = 'same')
    #4th convolutional layer commented out on tensorflow 

    #upscaling
    #transpose (undo pooling)
    #skip connections with downscaling layer, concatenate
    #convolve back down
    #5th upscaling layer commented out
    #unsure of padding for transpose layers, in tf it says same, but that isn't an option for torch
    self.transpose6 = nn.ConvTranspose2d(128, 64, 2, stride = 2)
    self.convolve6 = nn.Conv2d(128, 64, 3, padding = 'same') 
    self.transpose7 = nn.ConvTranspose2d(64, 32, 2, stride = 2)
    self.convolve7 = nn.Conv2d(64, 32, 3, padding = 'same')
    self.convout = nn.Conv2d(32, 1, 3, padding = 'same')

    #all the pooling and activation layers are the same, so we can just define them once
    self.pool = nn.MaxPool2d(2)
    self.act = nn.ReLU()
    self.norm = nn.BatchNorm2d(1)

  def forward(self, input):
    #initial convolution
    stacked_input = self.convin(input)
    stacked_input = self.act(stacked_input)
    
    #downscaling
    #convolve, pool
    conv1 = self.convolve1(stacked_input)
    conv1 = self.act(conv1)
    pool1 = self.pool(conv1)

    conv2 = self.convolve2(pool1)
    conv2 = self.act(conv2)
    pool2 = self.pool(conv2)

    conv3 = self.convolve3(pool2)
    conv3 = self.act(conv3)
    #4 and 5 are commented out of the original model
    #no pool here because the last downscaling layer gets upscaled after

    #upscaling
    #transpose, concat, convolve
    up6 = self.transpose6(conv3)
    #up6 = torch.cat((up6, conv2), 1) #dim 1 is the channel layer
    up6 = torch.cat((up6, conv2), len(up6.size()) - 3)
    #that change should make it not matter whether the image tensor includes batch size or not
    #the last 3 should always be: channel, height, width
    #this makes it less general potentially? but I think we will always be working with 2 dimensions for this project
    #if things stop working because there is another dimension added, this is where you would change first
    up6 = self.convolve6(up6)
    up6 = self.act(up6)
    
    up7 = self.transpose7(up6)
    #up7 = torch.cat((up7, conv1), 1)
    up7 = torch.cat((up7, conv1), len(up6.size()) - 3)
    up7 = self.convolve7(up7)
    up7 = self.act(up7)

    out = self.convout(up7)
    out = self.act(out)
    
    while out.dim() < 4:
      out = out.unsqueeze(0)

    #add batch normalization to fight exploding gradient problem
    out = self.norm(out)

    return torch.clamp(out, 0, 1)
  
  #prevent zero initialization
  def initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv2dTranspose):
          nn.init.xavier_normal_(m.weight)
          nn.init.constant_(m.bias, 0)
  
class convNet(nn.Module):
  def __init__(self, insize):
    super().__init__()

    self.insize = insize

    self.conv1 = nn.Conv2d(insize[0], 8, 3, padding = 'same')
    self.conv2 = nn.Conv2d(8, 16, 2, padding = 'same') #channels = 16, but with 2 pooling layers, so div by 16

    self.flat = nn.Flatten()

    self.lin1 = nn.Linear(insize[1]*insize[2], insize[1]*insize[2]/2)
    self.lin2 = nn.Linear(insize[1]*insize[2]/2, 36) #out to alphanumeric

    self.pool = nn.MaxPool2d(2)
    self.act = nn.ReLU()
    self.out = nn.Softmax()

  def forward(self, img):
    conv1 = self.conv1(img)
    conv1 = self.act(conv1)
    conv1 = self.pool(conv1)

    conv2 = self.conv2(conv1)
    conv2 = self.act(conv2)
    conv2 = self.pool(conv2)

    midpt = self.flat(conv2)

    lin1 = self.lin1(midpt)
    lin1 = self.act(lin1)

    lin2 = self.lin2(lin1)
    lin2 = self.out(lin2)

    return torch.argmax(lin2)
  
### TESTING MODELS ###
class testNet2(nn.Module):
  def __init__(self, insize):
    super().__init__()
    #resize from the input layer to the target size, done in the forward function

    #convolutional layer
    #kernel size = 8x8
    #padding = same
    #input size is the depth of the image, index 0 in the parameters
    self.conv1 = nn.Conv2d(insize[0], 1, 8, padding = 'same')
    self.conv2 = nn.Conv2d(1, 6, 8, padding = 'same')
    self.conv3 = nn.Conv2d(6, 3, 8, padding = 'same')
    self.conv4 = nn.Conv2d(3, 1, 8, padding = 'same')
    self.act = nn.ReLU()

  def forward(self, image):

    convolved1 = self.conv1(image)
    convolved1 = self.act(convolved1)

    convolved2 = self.conv2(convolved1)
    convolved2 = self.act(convolved2)

    convolved3 = self.conv3(convolved2)
    convolved3 = self.act(convolved3)

    convolved4 = self.conv4(convolved3)
    return self.act(convolved4)

class testNet(nn.Module):
  def __init__(self, insize):
    super().__init__()

    #format of function names: FunctionType Layer#
    self.convin = nn.Conv2d(insize[0], 1, 8, padding = 'same')
    self.act = nn.ReLU()
  
  def forward(self, image):
    convolve = self.convin(image)
    return self.act(convolve)