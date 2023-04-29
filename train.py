from dataset import plateDataset as pset
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
from Model import UNet

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
    model = UNet((params['batch_size'], 3, 1080, 1920)).to(train_device)

    #split dataset into training and testing
    newset = pset('data')

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
    train_loader = torch.utils.data.DataLoader(newset, **params)
    
    train_model(torch.optim.Adam(model.parameters(), lr=0.001), model, torch.nn.L1Loss(), 5, train_loader, train_device)


def display(imgs, names=['Input', 'True Mask', 'Predicted', 'Scaled Pred']):
    assert len(imgs) <= len(names)

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
    for images, masks in loader:
      images = images.to(device=device, dtype=torch.float32)
      masks = masks.to(device=device, dtype=torch.float32)
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

    #show the first image of the last batch at the end of the epoch
    showPred(model, img, msk)

    #save to out.npy
    #np.save('out.npy', y_pred.cpu().detach().numpy())

main()