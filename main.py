import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import argparse


from DataLoader import DataLoader, Batch
from Model import ConvNet
from Preprocessor import Preprocessor

def train(model: ConvNet, loader: DataLoader, early_stop: int = 25) -> None:
    #trains model
    epoch = 0
    train_loss_in_epoch = []
    average_train_loss = []
    
    preprocessor = Preprocessor()
    while(epoch <= 25):
        epoch += 1
        print('Epoch: ' , epoch)

        print('Training NN')
        loader.train_set()

        batch = 0
        while loader.has_next():
            batch = loader.get_next()
            batch = preprocessor.processBatch(batch)
            loss = model.train_batch(batch)
            print('Epoch: ', epoch , ' Batch: ' , batch , ' Loss: ' , loss)
            train_loss_in_epoch.append(loss)



        





def main():
    # main function
    print('loading data for training:')
    loader = DataLoader()
    model = ConvNet()
    train(model, loader) 



if __name__ == '__main__':
    main()




