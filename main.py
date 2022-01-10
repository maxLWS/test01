import torch
import numpy as np
import cv2
import os
from PIL import Image
from torchvision.io import read_image
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms,datasets
import matplotlib.pyplot as plt
import seaborn as sns
from torch.autograd import Variable
dataset_dir = './data/'             # 数据集路径
model_cp='./model/'
model_file='./model/model.pth'
'''数据处理'''
batch_size=32
tr="train"
te="test"
file_path=r"C:\\Users\\lws\\Desktop\\source\\python\\new_cat\\data"
transforms = transforms.Compose(
[

transforms.RandomResizedCrop(150),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
]
)
train_data = datasets.ImageFolder(os.path.join(file_path,tr), transforms)
test_data=datasets.ImageFolder(os.path.join(file_path,te), transforms)

train_loader=DataLoader(train_data,shuffle=True,batch_size=batch_size)
test_loader =DataLoader(test_data,batch_size=batch_size)

#模型设计
class RB(torch.nn.Module):
    def __init__(self,channels):
        super(RB, self).__init__()
        self.channels=channels
        self.conv1=torch.nn.Conv2d(channels,channels,kernel_size=3,padding=1)
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self,x):
        y=F.relu(self.conv1(x))
        y=self.conv2(y)
        return F.relu(x+y)

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1=torch.nn.Conv2d(3,20,5,5)
        self.conv2=torch.nn.Conv2d(20,50,4,1)
        self.mp=torch.nn.MaxPool2d(2)

        self.RB1=RB(20)
        self.RB2=RB(50)
        self.linear1=torch.nn.Linear(50*6*6,200)

        self.linear2 = torch.nn.Linear(200, 3)

    def forward(self,x):

        x=self.mp(F.relu(self.conv1(x)))
        x=self.RB1(x)
        x=self.mp(F.relu(self.conv2(x)))
        x=self.RB2(x)

        x=x.view(-1,50*6*6)
        x=self.linear1(x)
        x=self.linear2(x)
        return x
    def get_feature(self):
        # input = Variable(torch.randn(1, 3, 224, 224))
        input=self.process_image()

        x=input
        for index,layer in enumerate(self.pretrained_model):
            x=layer(x)
            if (index == self.selected_layer):
                return x

    def get_single_feature(self):
        features = self.get_feature()
        feature = features[:, 0, :, :]
        feature = feature.view(feature.shape[1], feature.shape[2])
        return feature

model=CNN()
cri=torch.nn.CrossEntropyLoss()
opt=optim.SGD(model.parameters(),lr=0.01,momentum=0.5)

def train(epoch):
    running_loss=0.0
    for batch_idx,data in enumerate(train_loader,0):
        inputs,target=data
        opt.zero_grad()

        outputs=model(inputs)
        loss=cri(outputs,target)
        loss.backward()
        opt.step()

        running_loss+=loss.item()
        if batch_idx%200==0:#输出200伦平均 loss
            print('[%d,%5d] loss:%0.3f'%(epoch+1,batch_idx+1,running_loss/200))
            running_loss=0.0

    torch.save(model.state_dict(),'{0}/model.pth'.format(model_cp))

def test():
    correct=0
    total=0
    with torch.no_grad():
        for data in test_loader:
            images,labels=data
            outputs=model(images)
            _,predicted=torch.max(outputs.data,dim=1)
            total +=labels.size(0)
            correct +=(predicted==labels).sum().item()
    print('Accuracy on test set : %d %%'%(100*correct/total))

'''def test_1():
    model = CNN()  # 实例化网络
    model.load_state_dict(torch.load(model_file))  # 加载训练好的模型参数
    model.eval()  # 设定为评估模式

    index = np.random.randint(0,test_loader.batch_size, 1)[0]  # 获取一个随机数，即随机从数据集中获取一个测试图片
    img = test_data.__getitem__(index)  # 获取一个图像

    img = Variable(img)  # 将数据放置在PyTorch的Variable节点中，并送入GPU中作为网络计算起点
    out = model(img)  # 网路前向计算，输出图片属于猫或狗的概率，第一列维猫的概率，第二列为狗的概率
    print(out)  # 输出该图像属于猫或狗的概率
    if out[0, 0] > out[0, 1]and out[0, 0] > out[0, 2]:  # 猫的概率大于狗
        print('the image is a cat')
    elif out[0, 1] > out[0, 0]and out[0, 1] > out[0, 2]:  # 猫的概率小于狗
        print('the image is a dog')
    elif out[0, 2] > out[0, 0] and out[0, 2] > out[0, 1]:
        print('the image is a flower')

    img = Image.open(test_loader.list_img[index])  # 打开测试的图片
    plt.figure('image')  # 利用matplotlib库显示图片
    plt.imshow(img)
    plt.show()
'''

if __name__=='__main__':
    for epoch in range(20):
        train(epoch)
        test()

