import torch.nn as nn
import torch

class SimpleConv(nn.Module):
    def __init__(self, key):
        #import pdb;pdb.set_trace() 
        super(SimpleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=key[3], out_channels=key[4], kernel_size=int(key[5][0]), stride=1, padding=1)
       
    def forward(self, x):
        x = self.conv1(x)
        return x

class SimpleLinear(nn.Module):
    def __init__(self, key):
        #import pdb;pdb.set_trace() 
        super(SimpleLinear, self).__init__()
        self.fc = nn.Linear(in_features=key[3], out_features=key[4])
       
    def forward(self, x):
        x = self.fc(x)
        return x

class SimpleRELU(nn.Module):
    def __init__(self, key):
        #import pdb;pdb.set_trace() 
        super(SimpleRELU, self).__init__()
        self.relu = nn.ReLU(inplace=True)
       
    def forward(self, x):
        x = self.relu(x)
        return x

class SimpleGELU(nn.Module):
    def __init__(self, key):
        #import pdb;pdb.set_trace() 
        super(SimpleGELU, self).__init__()
        self.gelu = nn.GELU(approximate='none')
       
    def forward(self, x):
        x = self.gelu(x)
        return x

class SimpleDropout(nn.Module):
    def __init__(self, key):
        #import pdb;pdb.set_trace() 
        super(SimpleDropout, self).__init__()
        self.drop = nn.Dropout(p=0.5, inplace=False)
       
    def forward(self, x):
        x = self.drop(x)
        return x

class SimpleAdaptPool(nn.Module):
    def __init__(self, key):
        #import pdb;pdb.set_trace() 
        super(SimpleAdaptPool, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size=key[2])
       
    def forward(self, x):
        x = self.pool(x)
        return x

class SimpleMaxPool(nn.Module):
    def __init__(self, key):
        #import pdb;pdb.set_trace() 
        super(SimpleMaxPool, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=int(key[5]), stride=int(key[6]), padding=int(key[7]), dilation=1, ceil_mode=False)
       
    def forward(self, x):
        x = self.pool(x)
        return x

class SimpleEncoder(nn.Module):
    def __init__(self, key):
        #import pdb;pdb.set_trace() 
        super(SimpleEncoder, self).__init__()
        self.encoder = nn.NonDynamicallyQuantizableLinear(in_features=int(key[5]), out_features=int(key[6]), bias=True)
       
    def forward(self, x):
        x = self.encoder(x)
        return x