import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import os

batch_size=128
IMG_SIZE=32
transform=transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])
train_dataset=datasets.MNIST(root='/data',download=True,train=True,transform=transform)
data_loader=DataLoader(train_dataset,shuffle=True,batch_size=batch_size,num_workers=os.cpu_count())