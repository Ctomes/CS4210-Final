from dataset import plateDataset as pset
from dataset import LetterSegDataset as segset
from charDataset import charDataset as cset

import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
from Model import UNet, testNet, testNet2, convNet
import torchvision

def main():
    #define device to use
    train_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    #setup dataloader
    train_set = cset('data')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0)
    print(np.shape(train_set[0][0]))

    #create the model
    model = convNet((3, 28, 28))
    model.to(train_device)

    train_model(torch.optim.Adam(model.parameters(), lr=1e-9), model, torch.nn.CrossEntropyLoss(), 150, train_loader, train_device)


def train_model(opt, model, loss_fn, epochs, loader, device):
    bank = dict()
    for i in range(65, 91):  # ASCII codes for uppercase A to Z
        bank[chr(i)] = i - 65
        #print(chr(i), ':', i - 65)
    
    for i in range(0, 10):
        bank[f'{i}'] = i + 26

    print(f'train device: {device}')
    for i in range(epochs):
        print(f"epoch: {i}")
        for n, combo in enumerate(loader):
            images, chars = combo
            #plt.imshow(np.transpose(np.asarray(images)[0], (1, 2, 0)))
            #plt.show()

            images = images.to(device=device, dtype=torch.float32)
            masks = chars.to(device=device, dtype=torch.float32).unsqueeze(0)
            #zero gradients
            opt.zero_grad()

            #forward
            y_pred = model(images)

            #evaluate loss
            loss = loss_fn(y_pred, chars.to(device)).to(device)

            #calculate gradient
            loss.backward()

            #step the optimizer
            opt.step()

            #print the loss
            print(f"loss: {loss.sum()}")

            #save the first image (there is probably a cleaner way to do this)
            img, msk = images[0], masks[0]
            if i == 0 and n % 10 == 0:
                print(f"predicted: {torch.argmax(y_pred)}, gt: {chars}")
        #show the first image of the last batch at the end of the epoch
        #showPred(model, img, msk)
        '''
        yhat = model(img).squeeze()
        if yhat.dim() < 3:
            yhat = yhat.unsqueeze(0)
        yhat = yhat.permute(1,2,0).detach().cpu()
        plt.imshow(np.asarray(yhat))
        plt.savefig(f'modelsTest/model3/outputmask{i}.png')

        img = img.permute(1,2,0).detach().cpu()
        plt.imshow(np.asarray(img))
        plt.savefig(f'modelsTest/model3/outputimg{i}.png')

        #save the model
        savepath = f'ModelWeights/Model3/UNet{i}.pt'
        torch.save(model.state_dict(), savepath)

        #save to out.npy
        #np.save(f'models/out{i}.npy', y_pred.cpu().detach().numpy())
        '''


if __name__ == '__main__':
    main()
