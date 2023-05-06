from read_class import readClass as read
from charDataset import charDataset as cset
import matplotlib.pyplot as plt

chars = cset('data')
print(len(chars))
for img, char in chars:
    print(char)
    plt.imshow(img)
    plt.show()
#for i in chars:
#    print(i)
'''
fulldict = read('data/frame000055.txt')
for char in fulldict[0].keys():
    if char.startswith('char'):
        print(fulldict[0][char])
        print(fulldict[0]['plate'][i])
        i += 1
'''

