from __future__ import print_function

import argparse
import csv
import os

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

import models
from utils_resnet import progress_bar

# Load Dataset
class ReflexDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)  # Load the CSV file
        self.img_dir = img_dir            # Path to the image directory
        self.transform = transform        # Image transformations

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get image file name and labels
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0])  # First column is the image name
        image = Image.open(f"{img_name}.png").convert("RGB")  # Open the image
        labels = self.data.iloc[idx, 1:].values.astype('float32')  # Remaining columns are labels

        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)

        return image, labels
    
# Paths to data
train_csv = "Reflex/labels_train.csv"
test_csv = "Reflex/labels_test.csv"
val_csv = "Reflex/labels_val.csv"
img_dir = "Reflex/reflex_img_1024_inter_nearest"

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--model', default="ResNet18", type=str,
                    help='model type (default: ResNet18)')
parser.add_argument('--name', default='0', type=str, help='name of run')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--batch-size', default=128, type=int, help='batch size')
parser.add_argument('--epoch', default=200, type=int,
                    help='total epochs to run')
# 当用户在命令行中输入 --no-augment 时，程序会将 augment 的值设为 False，从而禁用某些数据增强操作。
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='use standard augmentation (default: True)')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--alpha', default=1., type=float,
                    help='mixup interpolation coefficient (default: 1)')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()

torch.cuda.set_device(0)  # Set your desired GPU
torch.backends.cudnn.benchmark = True

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if args.seed != 0:
    torch.manual_seed(args.seed)


print('==> Preparing data..')
if args.augment:
    # 创建了一个用于训练图像的数据预处理管道。该管道包含了以下四个步骤
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 在图像中随机裁剪出大小为 32x32 的区域，并在裁剪前后分别填充 4 个像素。这个操作可以增加训练数据的多样性和鲁棒性。
        transforms.RandomHorizontalFlip(), # 随机对图像进行水平翻转。这个操作也可以增加数据的多样性。
        transforms.ToTensor(), # 转换为 PyTorch 中的 Tensor 数据类型。在转换过程中，像素值被缩放到 [0, 1] 之间。
        # 对图像进行归一化处理，将每个像素值减去均值，再除以标准差。这里给定了均值和标准差的值（RGB三个通道），这些值是在 ImageNet 数据集上计算出来的。
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

else:
    # 不进行数据增强
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

# 对测试集不进行加强，只进行基本预处理
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = ReflexDataset(csv_file=train_csv, img_dir=img_dir, transform=transform_train)

trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=args.batch_size,
                                          shuffle=True,
                                          num_workers=0 # 数据加载子进程数，加快加载速度
                                          )

testset = ReflexDataset(csv_file=test_csv, img_dir=img_dir, transform=transform_test)

testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=0)


# Model
if args.resume:
    # Load checkpoint.
    # 加载预训练模型
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7' + args.name + '_'
                            + str(args.seed))
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1
    rng_state = checkpoint['rng_state']
    torch.set_rng_state(rng_state)
else:
    print('==> Building model..')
    # 引入模型
    net = models.__dict__[args.model]()

if not os.path.isdir('results'):
    os.mkdir('results')
# 训练记录
logname = ('results/log_' + net.__class__.__name__ + '_' + args.name + '_'
           + str(args.seed) + '.csv')

if use_cuda:
 
    # 模型转移到GPU上
    net.cuda()
    # 多卡训练
    net = torch.nn.DataParallel(net)
    print(torch.cuda.device_count())
    # 启用了 PyTorch 的 CUDA 加速
    cudnn.benchmark = True
    print('Using CUDA..')

# 损失函数
criterion = torch.nn.BCEWithLogitsLoss()
# 优化器
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9,
                      weight_decay=args.decay)

# @parms: inputs, targets, args.alpha, use_cuda
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        # 生成一个服从 Beta 分布的随机数。该函数的输入参数 alpha 表示 Beta 分布的两个形状参数，通常都是正实数
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    # 使用随机排列生成一个索引 index，用于从输入数据 x 中选择一部分样本进行混合
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    # 使用 lam 对两个样本进行加权线性混合
    mixed_x = lam * x + (1 - lam) * x[index, :]

    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# 实现了 Mixup 数据增强方法的损失函数
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    # 该损失函数的含义是，对于每个样本，以 $\lambda$ 的概率使用标签 $y_a$ 计算损失函数，以 $(1-\lambda)$ 的概率使用标签 $y_b$ 计算损失函数。
    # pred是根据混合的训练数据x运算得到的
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    reg_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):

        # 数据转移到GPU
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        # 得到混合后的输入数据和label，权重等
        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets,
                                                       args.alpha, use_cuda)
       
        outputs = net(inputs)

        # mixup下的损失函数
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

        
        train_loss += loss.data
        predicted = (torch.sigmoid(outputs) > 0.5).float()  # Convert logits to binary predictions (128, 7)

        total += targets.size(0)

        correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                    + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar(batch_idx, len(trainloader),
                     'Loss: %.3f | Reg: %.5f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), reg_loss/(batch_idx+1),
                        100.*correct/total, correct, total))
        
        torch.cuda.empty_cache()

    return (train_loss/batch_idx, reg_loss/batch_idx, 100.*correct/total)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):

        
        if use_cuda:

            inputs, targets = inputs.cuda(), targets.cuda()

        outputs = net(inputs)
        loss = criterion(outputs, targets)
        
        test_loss += loss.data
        predicted = (torch.sigmoid(outputs) > 0.5).float()  # Convert logits to binary predictions (128, 7)

        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader),
                     'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss/(batch_idx+1), 100.*correct/total,
                        correct, total))
    acc = 100.*correct/total
    if epoch == start_epoch + args.epoch - 1 or acc > best_acc:
        checkpoint(acc, epoch)
    if acc > best_acc:
        best_acc = acc
    return (test_loss/batch_idx, 100.*correct/total)

# 保存模型权重
def checkpoint(acc, epoch):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net, # 模型
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt.t7' + args.name + '_'
               + str(args.seed))

# 调整学习率，第100轮和第150轮减小为十分之一
def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = args.lr
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':

    # 用csv记录
    if not os.path.exists(logname):
        with open(logname, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(['epoch', 'train loss', 'reg loss', 'train acc',
                                'test loss', 'test acc'])

    for epoch in range(start_epoch, args.epoch):
        train_loss, reg_loss, train_acc = train(epoch)
        test_loss, test_acc = test(epoch)
        adjust_learning_rate(optimizer, epoch)
        with open(logname, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([epoch, train_loss, reg_loss, train_acc, test_loss,
                                test_acc])


