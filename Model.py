import torch.nn as nn
import torch.nn.functional as F
class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=64,kernel_size=5,stride=2,padding=4)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.norm1 = nn.LocalResponseNorm(size=5)
        
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=5,padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.norm2 = nn.LocalResponseNorm(size=5)
        
        self.conv3 = nn.Conv2d(in_channels=128,out_channels=4,kernel_size=3)
        self.relu3 = nn.ReLU()
        
        self.flatten = nn.Flatten()
        
        self.fc4 = nn.Linear(816,1064)
        self.sig4 = nn.Sigmoid()
        
        self.fc5 = nn.Linear(1064,2048)
        self.sig5 = nn.Sigmoid()
        
        self.fc6 = nn.Linear(2048,3648) # change the TrainingSet class if you changed the size of the descriptor
        self.sig6 = nn.Sigmoid()
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.norm1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.norm2(x)
        
        x = self.conv3(x)
        x = self.relu3(x)
        
        x = self.flatten(x)
        
        x = self.fc4(x)
        x = self.sig4(x)
        
        x = self.fc5(x)
        x = self.sig5(x)
        
        x = self.fc6(x)
        x = self.sig6(x)
        return x