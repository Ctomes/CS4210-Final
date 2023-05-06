from dataset import plateDataset as pset
from dataset import LetterSegDataset as segset

import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
from Model import UNet, testNet, testNet2
import torchvision

#set device to GPU
def main():
    print(torch.version)
    #set parameters
    params = {'batch_size': 1,
              'shuffle': True,
              'num_workers': 6}
    
    #setup training device
    train_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    #train_device = 'cpu'
    print(f'using device: {train_device}')
    #model = UNet((3, 1080, 1920)).to(train_device)
    model = UNet((3, 144, 48))

    #split dataset into training and testing
    #newset = pset('data')
    newset = segset('data')
  
    #validate input and output shape and mask locations
    '''
    n, m = newset[0]
    print(n.min(), n.max())
    print(m.min(), m.max())
    print(np.shape(n))
    print(np.shape(m))
    n2 = np.transpose(n, (1, 2, 0))
    print(np.shape(n2))
    plt.imshow(n2)
    plt.show()
    plt.imshow(m)
    plt.show()
    '''

    #set up dataloader(s)
    train_loader = torch.utils.data.DataLoader(newset, batch_size=1, shuffle=True, num_workers=0)
    
    #train_model(torch.optim.Adam(model.parameters(), lr=1e-4), model, torchvision.ops.sigmoid_focal_loss, 150, train_loader, train_device)
    train_model2(torch.optim.Adam(model.parameters(), lr=1e-4), model, torchvision.ops.sigmoid_focal_loss, 150, train_loader, train_device)

    #save the model
    #savepath = 'ModelWeights/model1/UNetFinal.pt'
    savepath = 'ModelWeights/model2/UNetFinal.pt'
    torch.save(model.state_dict(), savepath)

def display(imgs, names=['Input', 'True Mask', 'Predicted', 'Scaled Pred']):
    assert len(imgs) <= len(names)

    print(imgs[2].size(), torch.max(imgs[2]), torch.min(imgs[2]))

    plt.figure(figsize=(15,15))
    for i in range(len(imgs)):
        plt.subplot(1,len(imgs), i+1)
        plt.title(names[i])
        if imgs[i].ndim==3:
            plt.imshow(imgs[i][:,:,0])
        else:
            plt.imshow(imgs[i])
        plt.axis('off')
    plt.show()

#assumes batch size is included. There is probably a better way but I couldn't find it rn
def showPred(model, img, mask):
  print(img.size())
  print(mask.size())
  yhat = model(img).squeeze() #since the batch size should always be 1
  #if the squeeze leaves you with a dimension less than 3, you know the channels were also 1
  if yhat.dim() < 3:
    yhat = yhat.unsqueeze(0)
  
  print(torch.max(yhat), torch.min(yhat))

  #move channels all to the end and remove gradients
  #for display preparation
  img = img.permute(1,2,0).detach().cpu()
  mask = mask.permute(1,2,0).detach().cpu()
  yhat = yhat.permute(1,2,0).detach().cpu()

  display([img, mask, yhat, torch.div(yhat, torch.max(yhat))])

# this training loop works if training with image masks
def train_model(opt, model, loss_fn, epochs, loader, device):
  print(f'train device: {device}')
  for i in range(epochs):
    print(f"epoch: {i}")
    for n, combo in enumerate(loader):
      images, masks = combo

      images = images.to(device=device, dtype=torch.float32)
      masks = masks.to(device=device, dtype=torch.float32).unsqueeze(0)
      #zero gradients
      opt.zero_grad()

      #forward
      y_pred = model(images)
      #evaluate loss
      loss = loss_fn(y_pred, masks.float()).to(device)
      #calculate gradient
      loss.sum().backward()

      #step the optimizer
      opt.step()

      #print the loss
      print(f"loss: {loss.sum()}")

      #save the first image (there is probably a cleaner way to do this)
      img, msk = images[0], masks[0]
      if n%80 == 0 and i == 0:
        showPred(model, img, msk)
    #show the first image of the last batch at the end of the epoch
    #showPred(model, img, msk)
    yhat = model(img).squeeze()
    if yhat.dim() < 3:
      yhat = yhat.unsqueeze(0)
    yhat = yhat.permute(1,2,0).detach().cpu()
    plt.imshow(np.asarray(yhat))
    plt.savefig(f'modelsTest/model1/outputmask{i}.png')

    img = img.permute(1,2,0).detach().cpu()
    plt.imshow(np.asarray(img))
    plt.savefig(f'modelsTest/model1/outputimg{i}.png')

    #save the model
    savepath = f'ModelWeights/Model1/UNet{i}.pt'
    torch.save(model.state_dict(), savepath)

    #save to out.npy
    #np.save(f'models/out{i}.npy', y_pred.cpu().detach().numpy())

# this training loop works when each item returns a set of images and masks (before batching)
def train_model2(opt, model, loss_fn, epochs, loader, device):
  print(f'train device: {device}')
  for i in range(epochs):
    print(f"epoch: {i}")
    for n, combo in enumerate(loader):
      imageset, maskset = combo
      imageset = [img.squeeze() for img in imageset]

      images = torch.stack(imageset, 0).to(device=device, dtype=torch.float32)
      masks = torch.stack(maskset, 0).to(device=device, dtype=torch.float32)
      
      #zero gradients
      opt.zero_grad()

      #forward
      y_pred = model(images)
      #evaluate loss
      loss = loss_fn(y_pred, masks.float()).to(device)
      #calculate gradient
      loss.sum().backward()

      #step the optimizer
      opt.step()

      #print the loss
      print(f"loss: {loss.sum()}")

      #save the first image (there is probably a cleaner way to do this)
      img, msk = images[0], masks[0]
      if n%80 == 0 and i == 0:
        showPred(model, img, msk)
    #show the first image of the last batch at the end of the epoch
    #showPred(model, img, msk)
    yhat = model(img).squeeze()
    if yhat.dim() < 3:
      yhat = yhat.unsqueeze(0)
    yhat = yhat.permute(1,2,0).detach().cpu()
    plt.imshow(np.asarray(yhat))
    plt.savefig(f'modelsTest/model2/outputmask{i}.png')

    img = img.permute(1,2,0).detach().cpu()
    plt.imshow(np.asarray(img))
    plt.savefig(f'modelsTest/model2/outputimg{i}.png')

    #save to out.npy
    #np.save(f'models/model2/out{i}.npy', y_pred.cpu().detach().numpy())

if __name__ == '__main__':
  main()
