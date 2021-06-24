import pandas as pd

import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from model import CatDog
from train_test import train, test
from args import park_parse_args

import os
from pathlib import Path
from time import time


def main():
    global args
    args = park_parse_args()

    torch.manual_seed(args.seed)

    data_folder = Path(args.data)
    train_folder = data_folder / 'train'
    test_folder = data_folder / 'test'
    checkpoints_folder = Path(args.checkpoint)

    transform = transforms.Compose([transforms.Resize((300, 400)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.25, 0.25, 0.25), (0.25, 0.25, 0.25))
                                    ]
                                   )

    train_data = datasets.ImageFolder(str(train_folder), transform=transform)
    # test_data = datasets.ImageFolder(str(test_folder), transform=transform)

    train_num = round(args.train_percentage * len(train_data))
    dev_num = len(train_data) - train_num

    train_set, dev_set = random_split(train_data, [train_num, dev_num])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=args.shuffle,
                              num_workers=1, pin_memory=True)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=args.shuffle,
                            num_workers=1, pin_memory=True)
    # test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=args.shuffle,
    #                          num_workers=1, pin_memory=True)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = CatDog().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    loss_func = nn.CrossEntropyLoss()

    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    best_test_acc = float('-inf')
    for epoch in range(1, args.epochs + 1):

        start = time()
        train(model, optimizer, epoch, device, train_loader, loss_func)
        end = time()
        print('Train time for epoch {}: {} seconds'.format(epoch, str(end - start)))

        start = time()
        test(model, epoch, device, train_loader, loss_func, 'train')
        end = time()
        print('Test time on train set for epoch {}: {} seconds'.format(
            epoch, str(end - start)))

        start = time()
        dev_acc = test(model, epoch, device, dev_loader, loss_func, 'dev')
        end = time()
        print('Test time on val set for epoch {}: {} seconds'.format(
            epoch, str(end - start)))

        scheduler.step()

        if dev_acc > best_test_acc:
            print('New best! {:.4f}%'.format(dev_acc * 100))
            with open(checkpoints_folder / 'model.pth', 'wb') as f:
                torch.save(model.state_dict(), f)
            best_test_acc = dev_acc

    # test(model, args.epochs, device, test_loader, loss_func, 'test')


if __name__ == '__main__':
    main()
