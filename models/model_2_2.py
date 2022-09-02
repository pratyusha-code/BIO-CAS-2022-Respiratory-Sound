import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchsummary import summary

class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.LeakyReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)

class Dense(nn.Module):
    def __init__(self, in_features, out_features, bias=True, activation = 'leakyrelu',  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dense_layer = nn.Linear(in_features, out_features)
        
        if activation == 'softmax':
            self.act = nn.Softmax(dim=1)
        elif activation == 'leakyrelu':
            self.act = nn.LeakyReLU()
        elif activation == 'tanh':
            self.act = nn.Tanh()
    
    def forward(self, x):
        out = self.dense_layer(x)
        return self.act(out)


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.image_classifier = nn.Sequential(

        	Conv2d(3, 3, kernel_size=3, stride=3, padding=1),
            Conv2d(3, 3, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(3, 3, kernel_size=3, stride=1, padding=1, residual=True),

        	Conv2d(3, 3, kernel_size=3, stride=3, padding=1),
            Conv2d(3, 3, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(3, 3, kernel_size=3, stride=1, padding=1, residual=True),

        	Conv2d(3, 5, kernel_size=3, stride=3, padding=1),
            Conv2d(5, 5, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(5, 5, kernel_size=3, stride=1, padding=1, residual=True),

        	Conv2d(5, 5, kernel_size=3, stride=3, padding=1),
            Conv2d(5, 5, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(5, 5, kernel_size=3, stride=1, padding=1, residual=True),

        	Conv2d(5, 7, kernel_size=3, stride=3, padding=1),
            Conv2d(7, 7, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(7, 7, kernel_size=3, stride=1, padding=1, residual=True),

            nn.Flatten(),

            Dense(7, 5, activation="softmax")

        )

    def forward(self, data):

        probabilities = self.image_classifier(data)
        return probabilities