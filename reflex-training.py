import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, random, torch, time, copy
import torch.nn.functional as F
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset
from PIL import Image, ImageOps, ImageFilter
from skimage.filters import threshold_local

# 数据集路径
dataset_path = './reflex_dataset'

# 定义类别
classes = ['normal', 'defect']  # RefleX数据集的两个类别

class XrayDataset(Dataset):
    """X-ray diffraction images dataset."""

    def __init__(self, root_dir, transform=None, phase=None):
        """
        Args:
            root_dir (string): 数据集根目录
            transform (callable, optional): 可选的图像转换
            phase (string): 训练或测试阶段
        """
        self.root_dir = root_dir
        self.transform = transform
        self.phase = phase
        
        # 读取数据集目录结构
        self.normal_path = os.path.join(root_dir, 'normal')
        self.defect_path = os.path.join(root_dir, 'defect')
        
        self.normal_files = os.listdir(self.normal_path)
        self.defect_files = os.listdir(self.defect_path)
        
        # 构建文件列表和标签
        self.all_files = []
        self.labels = []
        
        for f in self.normal_files:
            self.all_files.append(os.path.join(self.normal_path, f))
            self.labels.append(0)  # normal类别标签为0
            
        for f in self.defect_files:
            self.all_files.append(os.path.join(self.defect_path, f))
            self.labels.append(1)  # defect类别标签为1

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_path = self.all_files[idx]
        image = Image.open(img_path)
        label = self.labels[idx]
        
        sample = {'image': image, 'finding': label}
        
        if self.transform:
            sample = {'image': self.transform[self.phase](sample['image']), 
                     'finding': label}

        return sample

# 图像预处理类定义保持不变
class HistEqualization(object):
    def __call__(self, image):
        return ImageOps.equalize(image, mask=None)

class ContrastBrightness(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
    
    def __call__(self, image):
        image = np.array(image)
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                image[y,x] = np.clip(self.alpha*image[y,x] + self.beta, 0, 255)
        return Image.fromarray(np.uint8(image)*255)

class SmothImage(object):
    def __call__(self, image):
        return image.filter(ImageFilter.SMOOTH_MORE)

# 数据增强和标准化
data_transforms = {
    'train': transforms.Compose([
        transforms.Grayscale(1),
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        ContrastBrightness(1.2, 25),
        HistEqualization(),
        SmothImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.25])
    ]),
    'test': transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize(240),
        transforms.CenterCrop(224),
        ContrastBrightness(1.2, 25),
        HistEqualization(),
        SmothImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.25])
    ]),
}

# 创建数据集实例
image_datasets = {
    x: XrayDataset(
        root_dir=dataset_path,
        transform=data_transforms,
        phase=x)
    for x in ['train', 'test']
}

# 创建数据加载器
dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x],
        batch_size=16,
        shuffle=True,
        num_workers=8)
    for x in ['train', 'test']
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 训练记录列表
train_loss = []
train_acc = []
test_loss = []
test_acc = []

def train_model(model, criterion, optimizer, scheduler, num_epochs=25, device='cpu'):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for data in dataloaders[phase]:
                inputs = data['image']
                labels = data['finding']
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
            else:
                test_loss.append(epoch_loss)
                test_acc.append(epoch_acc)

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best validation Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model

# 加载预训练模型
model = models.resnet18(pretrained=True)

# 修改第一层以接收灰度图像
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

# 修改分类层
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(num_ftrs, num_ftrs),
    nn.Dropout(p=0.5),
    nn.Linear(num_ftrs, 2),  # 2个类别
)

# 设置所有参数可训练
for param in model.parameters():
    param.requires_grad = True

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# 训练模型
model_ft = train_model(model.to(device), criterion, optimizer_ft, exp_lr_scheduler,
                      num_epochs=50, device=device)

# 绘制训练过程
plt.style.use("ggplot")
plt.figure(figsize=(10, 6))
plt.plot(train_loss, label="train_loss")
plt.plot(test_loss, label="val_loss")
plt.plot(train_acc, label="train_acc")
plt.plot(test_acc, label="val_acc")
plt.title("Training Loss and Accuracy on RefleX Dataset")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.ylim(0, 1)
plt.legend(loc="lower left")
plt.savefig("training_plot.png")

# 评估模型
def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data in dataloader:
            images = data['image']
            labels = data['finding']
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return correct/total, all_preds, all_labels

# 计算每个类别的准确率
accuracy, predictions, true_labels = evaluate_model(model_ft, dataloaders['test'], device)
print(f'Overall Accuracy: {accuracy*100:.2f}%')

# 计算并显示混淆矩阵
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

cm = confusion_matrix(true_labels, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix.png')

# 打印详细的分类报告
print("\nClassification Report:")
print(classification_report(true_labels, predictions, target_names=classes))

# 保存模型
torch.save(model_ft, 'reflex_model.pkl')
