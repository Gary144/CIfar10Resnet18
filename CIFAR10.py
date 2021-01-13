import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

GPU = torch.cuda.is_available()
if not GPU:
    print ('You cannot use GPU')
else:
    print('GPU is ok')         #检查是否可以利用GPU

num_workers = 0                         #加载数据
batch_size = 16                         #每批加载16张图片
valid_size = 0.2                        # 测试集比例


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])                                      #将数据转换为torch.FloatTensor,并标准化


#训练集与测试集的数据
train_data = datasets.CIFAR10('data',
                              train=True,
                              download=True,
                              transform=transform)

test_data = datasets.CIFAR10('data',
                             train=True,
                             download=True,
                             transform=transform)


#获取验证集
#数据拆分
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int (np.floor(valid_size*num_train))
train_idx,valid_idx = indices[split:],indices[:split]

#define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

#perpare data loaders(combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size,
                                           sampler=train_sampler,num_workers=num_workers,pin_memory=True)
valid_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size,
                                           sampler=valid_sampler,num_workers=num_workers,pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_data,batch_size=batch_size,
                                          num_workers=num_workers,pin_memory=True)


classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']


# 定义卷积神经网络结构
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18():

   return ResNet(ResidualBlock)



model = ResNet(ResidualBlock)
print (model)

if GPU:
    model.cuda()


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.005)


n_epochs = 20

valid_loss_min = np.Inf #校准损失的跟踪变化

for epoch in range(1,n_epochs+1):

    #跟踪培训和验证的损失
    train_loss = 0.0
    valid_loss = 0.0

    ##################
    # 训练集的模型 #
    ##################
    model.train()
    for data,target in train_loader:
        #如果cuda可用，将张量移动到gpu
        if GPU:
            data,target = data.cuda(),target.cuda()
        #清除所有优化变量的渐变
        optimizer.zero_grad()
        #forward pass:通过将输入传递到模型来计算预测输出
        output = model(data)
        # calculate the batch loss
        loss = criterion(output,target)
        #后向传递:计算损失相对于模型参数的梯度
        loss.backward()
        #执行单个优化步骤(参数更新)
        optimizer.step()
        #updata培训损失
        train_loss += loss.item()*data.size(0)

    ###############
    # 验证集模型 #
    ##################
    model.eval()
    for data,target in valid_loader:
        if GPU:
            data,target = data.cuda(),target.cuda()
        output = model(data)
        loss = criterion(output,target)
        valid_loss += loss.item()*data.size(0)

    #计算平均损失
    train_loss = train_loss/len(train_loader.sampler)
    valid_loss = valid_loss/len(valid_loader.sampler)

    #显示训练集与验证集的损失函数
    print('Epoch:{} \tTraining loss:{} \tValidation loss:{}'.format(
        epoch,train_loss,valid_loss
    ))

    #如果验证集损失函数减少，就保存模型
    if valid_loss <= valid_loss_min:
        print ('Validation loss decreased ({} --> {}). Saving model ...'.format(
            valid_loss_min,valid_loss
        ))
        torch.save(model.state_dict(),'model_cifar.pt')
        valid_loss_min = valid_loss

model.load_state_dict(torch.load('model_cifar.pt',map_location=torch.device('cpu')))

# 跟踪测试损失
test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

model.eval()
# 迭代测试数据
for data, target in test_loader:
    # 如果CUDA可用，将张量移动到GPU
    if GPU:
        data, target = data.cuda(), target.cuda()
    # forward pass:通过将输入传递到模型来计算预测输出
    output = model(data)
    # 计算批次损失
    loss = criterion(output, target)
    # 更新测试损失
    test_loss += loss.item()*data.size(0)
    # 将输出概率转换为预测类
    _, pred = torch.max(output, 1)
    # 比较预测和真实标签
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not GPU else np.squeeze(correct_tensor.cpu().numpy())
    # 计算每个对象类的测试精度
    for i in range(batch_size):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

#求出平均误差
test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))


