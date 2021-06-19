from collections import OrderedDict
import torch
import torch.optim as optim
import torchvision
from torchvision import transforms
import os
import resnet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

img_shape = (3,32,32)
train_batch_size = 256
test_batch_size = 10

#data_dir = "~/research/vision/data/cifar100"

def build_model(model_load_path):
  
  learning_rate = 1e-3  
  checkpoint_path = model_load_path

  model = resnet.resnet50()

  optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  model = model.to(device=device)

  return model, optimizer

def get_data_loaders(data_dir):

  transformation = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

  train_dataset = torchvision.datasets.CIFAR100(
      root=data_dir, train=True, transform=transformation, download=True
  )

  train_loader = torch.utils.data.DataLoader(
      train_dataset, batch_size=train_batch_size, shuffle=True
  )

  test_dataset = torchvision.datasets.CIFAR100(
    root=data_dir, train=False, transform=transformation, download=True
  )

  test_loader = torch.utils.data.DataLoader(
      test_dataset, batch_size=test_batch_size, shuffle=False
  )

  return train_loader, test_loader

