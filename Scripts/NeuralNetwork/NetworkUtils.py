import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def create_data_loader(data_root, batch_size, workers):
    dataset = datasets.ImageFolder(root=data_root,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
    # dataset = datasets.ImageFolder(root=data_root,
    #                  transform=transforms.Compose([
    #                      transforms.Resize(64),
    #                      transforms.CenterCrop(64),
    #                      transforms.ToTensor(),
    #                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #                  ]))
    # Create the dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    return dataloader
