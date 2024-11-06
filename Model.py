from typing import List, Tuple

import numpy as np
import torch.nn as nn
import torch
from DataLoader import Batch

from torch.nn import functional as F

class Block(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1):
        super(Block, self).__init__()
        
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        
        # This block contains a convolutional layer
        # then a batch normalization layer
        
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride)
        self.bn = nn.BatchNorm2d(out_channel)
        return
    
    def forward(self, x):
        # passes the input image through a convolutional layer
        # followed by a batch normalization layer and relu transition
        out = F.relu(self.bn(self.conv(x)))
        return out


class ConvNet(nn.Module):
    def __init__(self) -> None:
        super(ConvNet, self).__init__()
        self.snap_ID = 0
        self.block1 = Block(1, 32, 5, 2)
        self.block2 = Block(32, 64, 5, 2)
        self.block3 = Block(64, 128, 3, 1)
        self.block4 = Block(128, 128, 3, 1)
        self.block5 = Block(128, 256, 3, 1)
        #five convolutional layers and then one pooling layer
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        
        self.loss = nn.CTCLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.1)

        #linear layer for averaging predictions
        self.fc = nn.Linear(256, 126)


    def train_batch(self, batch: Batch) -> float:
        num_batch_elements = len(batch.img)
        running_loss = 0.0

        if torch.cuda.is_available():
            print('Training batch on GPU')
            batch = batch.cuda()

        target_lengths = [len(batch.gt_text[x]) for x in range(num_batch_elements)]
        target_tensor = torch.tensor(target_lengths, dtype = torch.long)
        #target_tensor = target_tensor.to(torch.long)


        input_lengths = [len(batch.img[x]) for x in range(num_batch_elements)]
        print("input lengths: ", input_lengths)
        input_lengths = torch.tensor(input_lengths, dtype = torch.long)
        #input_tensor = input_tensor.to(torch.long)



    
        for i in range(num_batch_elements):
            #zeros optimizer gradients
            self.optimizer.zero_grad()
            

            input_tensor = torch.from_numpy(batch.img[i]).float()   
            input_tensor = input_tensor.view(1, 1, 150, 2000)         
            logits = self.forward(input_tensor)
            logits = logits.reshape(logits.size()[0], 5, 1)
            # batch.gt_text[i] = "A MOVE by some guy" -> ascii array

            # UNLIKELY TO WORK
            truths = batch.gt_text[i]
            array_of_gt_text = [ord(x) for x in truths]
            truth = np.asarray(array_of_gt_text)
            truth = torch.from_numpy(truth).float()
            

            #loss = self.loss(logits, torch.Tensor.new_tensor(batch.gt_text[i]))

            loss = self.loss(logits, truth, input_lengths, target_tensor)

            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()

        return running_loss / num_batch_elements





    def forward(self, x):
        #batch_size = x.size(0)
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        
        # .squeeze() operation remove unnecessary dimension
        # i.e if X is of shape (128, 128, 1, 1)
        # X.unsqueeze() becomes (128, 128)
        out = self.avgpool(out).squeeze()
        out = self.fc(out)
        return out

    def save(self) -> None:
        #Save model to file.
        path = 'C:\\Users\\plano\\Documents\\1-SCHOOL STUFF\\2024-2025 Year 3\\Hackathon OHIO 2024\\Load\\model.pth'
        torch.save(self.state_dict(), path)
        print("model saved in " + path)



    def load(self) -> None:
        path = 'C:\\Users\\plano\\Documents\\1-SCHOOL STUFF\\2024-2025 Year 3\\Hackathon OHIO 2024\\Load\\model.pth'
        print('Loading NN from ' + path)
        self.load_state_dict(torch.load(path))








        





