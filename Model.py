import torch.nn as nn
import torch


class UNet(nn.Module):
    def __init__(self, inshape):
        super().__init__()

        self.inshape = inshape

    #downscaling
        self.convolve1 = nn.Conv2d(inshape[1], 16, 3, padding = 'same')
        self.convolve2 = nn.Conv2d(16, 32, 4, padding = 'same')
        self.convolve3 = nn.Conv2d(32, 64, 3, padding = 'same')
        #self.convolve4 = nn.Conv2d(64, 128, 3, padding = 'same')

        #upscaling
        #transpose (undo pooling)
        #skip connections with downscaling layer, concatenate
        #convolve back down
        #self.transpose6 = nn.ConvTranspose2d(128, 64, 2, stride = 2)
        #self.convolve6 = nn.Conv2d(128, 64, 3, padding = 'same') 
        self.transpose7 = nn.ConvTranspose2d(64, 32, 2, stride = 2)
        self.convolve7 = nn.Conv2d(64, 32, 3, padding = 'same')
        self.transpose8 = nn.ConvTranspose2d(32, 16, 2, padding = 'same')
        self.convolve8 = nn.Conv2d(32, 16, 3, padding = 'same')
        self.convout = nn.Conv2d(16, 1, 3, padding = 'same')

        #all the pooling and activation layers are the same, so we can just define them once
        self.pool = nn.MaxPool2d(2)
        self.act = nn.ReLU()

        #normalization because gradient tends to explode with u-nets
        self.norm = nn.BatchNorm2d(1)

    def forward(self, image):
      #downscaling
      down1 = self.convolve1(image)
      down1 = self.act(down1)
      pool1 = self.pool(down1)

      down2 = self.convolve2(pool1)
      down2 = self.act(down2)
      pool2 = self.pool(down2)

      down3 = self.convolve3(pool2)
      down3 = self.act(down3)
      #no pool, this gets upscaled immediately

      #upscaling
      #Transpose, concat, convolve
      up1 = self.transpose7(down3)
      up1 = torch.cat((up1, down2), (len(up1.size()) - 3)) #skip connection, concatenate at the channel layer
      up1 = self.convolve7(up1)
      up1 = self.act(up1)

      up2 = self.transpose8(up1)
      up2 = torch.cat((up2, down1), (len(up2.size()) - 3)) #skip connection, concat with the channel layer
      up2 = self.convolve8(up2)
      up2 = self.act(up2)

      out = self.convout(up2)
      out = self.act(out)

      #ensure that the output has the correct number of dimensions
      while out.dim() < 4:
        out = out.unsqueeze(0)

      #batch normalization to prevent exploding gradient
      return self.norm(out)