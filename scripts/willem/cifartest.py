import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from datarep.matplotlibsettings import *

from biasadaptation.utils import utils

import helperfuncs

randgen1 = torch.Generator()
randgen1.manual_seed(5)
randgen2 = torch.Generator()
randgen2.manual_seed(5)

print('==> Preparing data..')
transform_train1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    helperfuncs.ReshapeTransform((-1,))
])

transform_train2 = transforms.Compose([
    transforms.ToTensor(),
])


trainset1 = helperfuncs.get_dataset("CIFAR10", rotate=False)
trainloader1 = torch.utils.data.DataLoader(
    trainset1, batch_size=1, shuffle=True, generator=randgen1)

trainset2 = torchvision.datasets.CIFAR10(
    root="/Users/wybo/Data/", train=True, download=True, transform=transform_train2)
trainloader2 = torch.utils.data.DataLoader(
    trainset2, batch_size=1, shuffle=True, generator=randgen2)


iter2 = iter(trainloader2)

for batch_idx, (inputs, targets) in enumerate(trainloader1):
    inputs_, targets_ = next(iter2)

    print(batch_idx)
    print(inputs.shape)
    print(targets)



    img = inputs[0]
    # img = torch.reshape(img, (3,32,32))
    img = utils.to_image_cifar(img)
    print('---', img.shape)
    img_ = inputs_[0]

    # avgs = (0.4914, 0.4822, 0.4465)
    # stds = (0.2023, 0.1994, 0.2010)
    # for idx in range(3):
    #     img_[idx] = (img_[idx] - avgs[idx]) / stds[idx]

    print(torch.norm(img[0]))
    print(torch.norm(img[1]))
    print(torch.norm(img[2]))

    print(img.shape)

    pl.figure()
    ax = pl.subplot(121)
    ax.imshow(transforms.ToPILImage()(img))
    ax = pl.subplot(122)
    ax.imshow(transforms.ToPILImage()(img_))
    pl.show()
