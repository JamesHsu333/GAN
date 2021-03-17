import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.main = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )

    def forward(self, inputs):
        return self.main(inputs)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        return self.main(inputs)

def d_loss_fn(inputs, targets):
    return nn.BCELoss()(inputs, targets)

def g_loss_fn(inputs):
    targets = torch.ones([inputs.shape[0], 1, 1, 1])
    targets = targets.to('cuda:0')
    return nn.BCELoss()(inputs, targets)

if __name__ =='__main__':
    d = Discriminator().cuda()
    summary(d, (1, 28, 28))
    g = Generator().cuda()
    summary(g, (64, 1, 1))