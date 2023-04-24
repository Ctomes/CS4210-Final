from dataset import plateDataset as pset
import matplotlib.pyplot as plt

#testing out dataset functionality
newset = pset('data')

for n in newset:
    plt.imshow(n[0])
    plt.show()