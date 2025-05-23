#%%

import os
import numpy as np

path = '/Users/VAND/Documents/TEACHING/02506/data/bugNIST2D/'  # path to unzipped data directory
train_filenames = sorted(os.listdir(path + 'train')) 
train_targets = np.loadtxt(path + 'train_targets.txt', dtype=int) 


# %%
import matplotlib.pyplot as plt
import PIL.Image

rg = np.random.default_rng()

class_names = [ 
    'AC: brown cricket', 'BC: black cricket', 'BF: blow fly', 
    'BL: buffalo beetle larva', 'BP: blow fly pupa',  'CF: curly-wing fly', 
    'GH: grasshopper', 'MA: maggot', 'ML: mealworm', 
    'PP: green bottle fly pupa',  'SL: soldier fly larva',  'WO: woodlice'
    ]

for i in range(12):

    fig, ax = plt.subplots(1, 8, figsize=(15, 5))

    this_class = np.where(train_targets == i)[0]
    random_subset = sorted(rg.choice(this_class, 8, replace=False))

    for j in range(8):
        
        filename = train_filenames[random_subset[j]]
        image = PIL.Image.open(path + 'train/' + filename)

        ax[j].imshow(image)
        ax[j].set_title(filename) 
    
    fig.suptitle(f'Class {i} ({class_names[i]})')
    plt.show()


# %%
