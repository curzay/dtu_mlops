import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import pandas as pd
from torch.utils.data import Dataset



def mnist():

    def concat_npz(path):
        # Function to concatenate 5 training npz files into one
        images = []
        labels = []
        for file in range(0,5):
            with np.load(path+'\\train_'+str(file)+'.npz') as data1:
                img1 = data1['images']
                for img11 in img1:
                    images.append(img11)
                lab1 = data1['labels']
                for lab11 in lab1:
                    labels.append(lab11)

        # Save the concatenated lists to a new .npz file
        np.savez_compressed(r'C:\Users\carol\Documents\1ºAI\dtu_mlops-main\data\corruptmnist\train.npz', images=np.array(images), labels=np.array(labels))


    class NpzDataset(torch.utils.data.Dataset):
        # Class to create a torch Dataset class for inputting into Dataloader
        def __init__(self, data):
            self.img = data['images']
            self.lab = data['labels']
        
        def __getitem__(self, idx):
            return (self.img[idx], self.lab[idx])
        
        def __len__(self):
            return len(self.img)


    # Load train and test sets
    concat_npz(r'C:\Users\carol\Documents\1ºAI\dtu_mlops-main\data\corruptmnist')
    data_train = np.load(r'C:\Users\carol\Documents\1ºAI\dtu_mlops-main\data\corruptmnist/train.npz')
    data_test = np.load(r'C:\Users\carol\Documents\1ºAI\dtu_mlops-main\data\corruptmnist/test.npz')

    # Create Datasets from the npz files
    dataset_train = NpzDataset(data_train)
    dataset_test = NpzDataset(data_test)

    # Create dataloaders from the Datasets
    trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(dataset_test, batch_size=64, shuffle=True)
        
    return trainloader, testloader

mnist()