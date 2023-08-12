import cv2
import os
import random
import zipfile
import numpy as np
from copy import deepcopy
from PIL import Image, ImageEnhance
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap as LSC

import paddle
from paddle import nn
from paddle.metric import Metric
from paddle.framework import ParamAttr
from paddle.io import DataLoader, Dataset
from paddle.nn import initializer as I, functional as F
from paddle.optimizer import Adam
from paddle.optimizer.lr import CosineAnnealingDecay

#设置超参数
EPOCHS = 10               # 模型训练的总轮数
BATCH_SIZE = 8           # 每批次的样本数

N_CLASSES = 2            # 图像分类种类数量
IMG_SIZE = (256, 256)    # 图像缩放尺寸

SRC_PATH = "./data/data69911/BuildData.zip"  # 压缩包路径
DST_PATH = "./data"                          # 解压路径
DATA_PATH = {                                # 数据集路径
    "img": DST_PATH + "/image",    # 正常图像
    "lab": DST_PATH + "/label",    # 分割图像
}
INFER_PATH = {                               # 预测数据集路径
    "img": ["./work/1.jpg", "./work/2.jpg"],   # 正常图像
    "lab": ["./work/1.png", "./work/2.png"],   # 分割图像
}
MODEL_PATH = "UNet3+.pdparams"               # 模型参数保存路径

#数据准备
#解压数据库
if not os.path.isdir(DATA_PATH["img"]) or not os.path.isdir(DATA_PATH["lab"]):
    z = zipfile.ZipFile(SRC_PATH, "r")   # 以只读模式打开zip文件
    z.extractall(path=DST_PATH)          # 解压zip文件至目标路径
    z.close()
print("The dataset has been unpacked successfully!")

#划分数据库
train_list, test_list = [], []         # 存放图像路径与标签路径的映射
images = os.listdir(DATA_PATH["img"])  # 统计数据集下的图像文件

for idx, img in enumerate(images):
    lab = os.path.join(DATA_PATH["lab"], img.replace(".jpg", ".png"))
    img = os.path.join(DATA_PATH["img"], img)
    if idx % 10 != 0:                  # 按照1:9的比例划分数据集
        train_list.append((img, lab))
    else:
        test_list.append((img, lab))

#数据增强
def random_brightness(img, lab, low=0.5, high=1.5):
    x = random.uniform(low, high)
    img = ImageEnhance.Brightness(img).enhance(x)
    return img, lab

def random_contrast(img, lab, low=0.5, high=1.5):
    x = random.uniform(low, high)
    img = ImageEnhance.Contrast(img).enhance(x)
    return img, lab

def random_color(img, lab, low=0.5, high=1.5):
    x = random.uniform(low, high)
    img = ImageEnhance.Color(img).enhance(x)
    return img, lab

def random_sharpness(img, lab, low=0.5, high=1.5):
    x = random.uniform(low, high)
    img = ImageEnhance.Sharpness(img).enhance(x)
    return img, lab

def random_rotate(img, lab, low=0, high=360):
    angle = random.choice(range(low, high))
    img, lab = img.rotate(angle), lab.rotate(angle)
    return img, lab

def random_flip(img, lab, prob=0.5):
    img, lab = np.asarray(img), np.asarray(lab)
    if random.random() < prob:
        img, lab = img[:, ::-1, :], lab[:, ::-1]
    if random.random() < prob:
        img, lab = img[::-1 , :, :], lab[::-1 , :]
    img, lab = Image.fromarray(img), Image.fromarray(lab)
    return img, lab

def random_noise(img, lab, low=0, high=10):
    img = np.asarray(img)
    sigma = np.random.uniform(low, high)
    noise = np.random.randn(img.shape[0], img.shape[1], 3) * sigma
    img = img + np.round(noise).astype('uint8')
    # 将矩阵中的所有元素值限制在0~255之间：
    img[img > 255], img[img < 0] = 255, 0
    img = Image.fromarray(img)
    return img, lab

def image_augment(img, lab, prob=0.5):
    opts = [random_brightness, random_contrast, random_color, random_flip, random_noise, random_rotate, random_sharpness,]
    for func in opts:
        if random.random() < prob:
            img, lab = func(img, lab)
    return img, lab

#数据预处理
class MyDataset(Dataset):
    def __init__(self, label_list, transform, augment=None):
        super(MyDataset, self).__init__()
        random.shuffle(label_list)       # 打乱映射列表
        self.label_list = label_list
        self.transform = transform
        self.augment = augment

    def __getitem__(self, index):
        img_path, lab_path = self.label_list[index]
        img, lab = self.transform(img_path, lab_path, self.augment)
        return img, lab

    def __len__(self):
        return len(self.label_list)


def data_mapper(img_path, lab_path, augment=None):
    img = Image.open(img_path).convert("RGB")
    lab = cv2.cvtColor(cv2.imread(lab_path), cv2.COLOR_RGB2GRAY)
    # 将标签文件进行灰度二值化：
    _, lab = cv2.threshold(src=lab,
                           thresh=170,
                           maxval=255,
                           type=cv2.THRESH_BINARY_INV)
    lab = Image.fromarray(lab).convert("L")
    # 将图像缩放为IMG_SIZE大小的高质量图像：
    img = img.resize(IMG_SIZE, Image.ANTIALIAS)
    lab = lab.resize(IMG_SIZE, Image.ANTIALIAS)
    if augment is not None:    # 数据增强
        img, lab = augment(img, lab)
    # 将图像转为numpy数组，并转换图像的格式：
    img = np.array(img).astype("float32").transpose((2, 0, 1))
    lab = np.array(lab).astype("int32")
    # 将图像数据归一化，并转换成Tensor格式：
    img = paddle.to_tensor(img / 255.0)
    lab = paddle.to_tensor(lab // 255)
    lab = paddle.unsqueeze(lab,0)
    return img, lab

train_dataset = MyDataset(train_list, data_mapper, image_augment)  # 训练集
test_dataset = MyDataset(test_list, data_mapper, augment=None)     # 测试集

img, lab=train_dataset[0]
print(lab.shape)
print(img.shape)

#定义数据提供器
train_loader = DataLoader(train_dataset,
                          batch_size=BATCH_SIZE,
                          num_workers=2,
                          shuffle=True,
                          drop_last=False)
test_loader = DataLoader(test_dataset,
                         batch_size=BATCH_SIZE,
                         num_workers=2,
                         shuffle=False,
                         drop_last=False)

#网络配置
#定义网络结构
import paddle.nn as nn

class selfunet(nn.Layer):
    def __init__(self, num_classes):
        super().__init__()
        #下采样
        self.conv3_64 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3,padding='same')
        self.bn64_1=nn.BatchNorm(64,act='relu')
        #batchnorm层作用：保证了每一次数据归一化后还保留有之前学习来的特征分布，同时又能完成归一化的操作，加速训练。
        self.conv64_64 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3,padding='same')
        self.bn64_2=nn.BatchNorm(64,act='relu')
        self.maxpool1 = nn.MaxPool2D(kernel_size=2,stride=2)

        self.conv64_128 = nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3,padding='same')
        self.bn128_1=nn.BatchNorm(128,act='relu')
        self.conv128_128 = nn.Conv2D(in_channels=128, out_channels=128, kernel_size=3,padding='same')
        self.bn128_2=nn.BatchNorm(128,act='relu')
        self.maxpool2 = nn.MaxPool2D(kernel_size=2,stride=2)

        self.conv128_256 = nn.Conv2D(in_channels=128, out_channels=256, kernel_size=3,padding='same')
        self.bn256_1=nn.BatchNorm(256,act='relu')
        self.conv256_256 = nn.Conv2D(in_channels=256, out_channels=256, kernel_size=3,padding='same')
        self.bn256_2=nn.BatchNorm(256,act='relu')
        self.maxpool3 = nn.MaxPool2D(kernel_size=2,stride=2)

        self.conv256_512 = nn.Conv2D(in_channels=256, out_channels=512, kernel_size=3,padding='same')
        self.bn512_1=nn.BatchNorm(512,act='relu')
        self.conv512_512 = nn.Conv2D(in_channels=512, out_channels=512, kernel_size=3,padding='same')
        self.bn512_2=nn.BatchNorm(512,act='relu')
        self.maxpool4 = nn.MaxPool2D(kernel_size=2,stride=2)

        self.conv512_1024 = nn.Conv2D(in_channels=512, out_channels=1024, kernel_size=3,padding='same')
        self.bn1024_1=nn.BatchNorm(1024,act='relu')
        self.conv1024_1024 = nn.Conv2D(in_channels=1024, out_channels=1024, kernel_size=3,padding='same')
        self.bn1024_2=nn.BatchNorm(1024,act='relu')

        #上采样
        self.upsample1 = nn.Upsample(scale_factor=2.0)
        self.upconv1 = nn.Conv2D(in_channels=1024,out_channels=512,kernel_size=1,padding='SAME')
        self.bn512_3 = nn.BatchNorm(512,act='relu')
        self.conv1024_512 = nn.Conv2D(in_channels=1024,out_channels=512,kernel_size=3,padding='SAME')
        self.bn512_4 = nn.BatchNorm(512,act='relu')
        #注意self.conv1024_512该卷积，输入为1024，其实此时有一个级联过程（即上采样得到的蓝色与之前下采样得到的白色级联），在前向传播时实现。
        self.conv512_512_2 = nn.Conv2D(in_channels=512,out_channels=512,kernel_size=3,padding='SAME')
        self.bn512_5 = nn.BatchNorm(512,act='relu')

        self.upsample2 = nn.Upsample(scale_factor=2.0)
        self.upconv2 = nn.Conv2D(in_channels=512,out_channels=256,kernel_size=1,padding='SAME')
        self.bn256_3 = nn.BatchNorm(256,act='relu')
        self.conv512_256 = nn.Conv2D(in_channels=512,out_channels=256,kernel_size=3,padding='SAME')
        self.bn256_4 = nn.BatchNorm(256,act='relu')
        self.conv256_256_2 = nn.Conv2D(in_channels=256,out_channels=256,kernel_size=3,padding='SAME')
        self.bn256_5 = nn.BatchNorm(256,act='relu')

        self.upsample3 = nn.Upsample(scale_factor=2.0)
        self.upconv3 = nn.Conv2D(in_channels=256,out_channels=128,kernel_size=1,padding='SAME')
        self.bn128_3 = nn.BatchNorm(128,act='relu')
        self.conv256_128 = nn.Conv2D(in_channels=256,out_channels=128,kernel_size=3,padding='SAME')
        self.bn128_4 = nn.BatchNorm(128,act='relu')
        self.conv128_128_2 = nn.Conv2D(in_channels=128,out_channels=128,kernel_size=3,padding='SAME')
        self.bn128_5 = nn.BatchNorm(128,act='relu')

        self.upsample4 = nn.Upsample(scale_factor=2.0)
        self.upconv4 = nn.Conv2D(in_channels=128,out_channels=64,kernel_size=1,padding='SAME')
        self.bn64_3 = nn.BatchNorm(64,act='relu')
        self.conv128_64 = nn.Conv2D(in_channels=128,out_channels=64,kernel_size=3,padding='SAME')
        self.bn64_4 = nn.BatchNorm(64,act='relu')
        self.conv64_64_2 = nn.Conv2D(in_channels=64,out_channels=64,kernel_size=3,padding='SAME')
        self.bn64_5 = nn.BatchNorm(64,act='relu')
        self.cls = nn.Conv2D(in_channels=64,out_channels=num_classes,kernel_size=3,stride=1,padding=1)

    #传播
    def forward(self, x):
        logit_lists=[]#最后的输出
        short_cuts=[]#记录下采样的结果，后续实现级联
        #Encoder
        x=self.conv3_64(x)
        x=self.bn64_1(x)
        x=self.conv64_64(x)
        x=self.bn64_2(x)
        short_cuts.append(x)
        x=self.maxpool1(x)

        x=self.conv64_128(x)
        x=self.bn128_1(x)
        x=self.conv128_128(x)
        x=self.bn128_2(x)
        short_cuts.append(x)
        x=self.maxpool2(x)

        x=self.conv128_256(x)
        x=self.bn256_1(x)
        x=self.conv256_256(x)
        x=self.bn256_2(x)
        short_cuts.append(x)
        x=self.maxpool3(x)

        x=self.conv256_512(x)
        x=self.bn512_1(x)
        x=self.conv512_512(x)
        x=self.bn512_2(x)
        short_cuts.append(x)
        x=self.maxpool4(x)

        x=self.conv512_1024(x)
        x=self.bn1024_1(x)
        x=self.conv1024_1024(x)
        x=self.bn1024_2(x)

        #Decoder
        x=self.upsample1(x)
        x=self.upconv1(x)
        x=self.bn512_3(x)
        x=paddle.concat([x,short_cuts[-1]],axis=1) #级联
        x=self.conv1024_512(x)
        x=self.bn512_4(x)
        x=self.conv512_512_2(x)
        x=self.bn512_5(x)

        x=self.upsample2(x)
        x=self.upconv2(x)
        x=self.bn256_3(x)
        x=paddle.concat([x,short_cuts[-2]],axis=1) #级联
        x=self.conv512_256(x)
        x=self.bn256_4(x)
        x=self.conv256_256_2(x)
        x=self.bn256_5(x)

        x=self.upsample3(x)
        x=self.upconv3(x)
        x=self.bn128_3(x)
        x=paddle.concat([x,short_cuts[-3]],axis=1) #级联
        x=self.conv256_128(x)
        x=self.bn128_4(x)
        x=self.conv128_128_2(x)
        x=self.bn128_5(x)

        x=self.upsample4(x)
        x=self.upconv4(x)
        x=self.bn64_3(x)
        x=paddle.concat([x,short_cuts[-4]],axis=1) #级联
        x=self.conv128_64(x)
        x=self.bn64_4(x)
        x=self.conv64_64_2(x)
        x=self.bn64_5(x)

        logit=self.cls(x) #得到结果，2*256*256
        logit_lists.append(logit)

        return logit_lists

model = selfunet(2)
paddle.Model(model).summary((BATCH_SIZE, 3) + IMG_SIZE)  # 可视化模型结构

#定义acc
class Selfacc(Metric):
    def __init__(self, name='selfacc', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tp = 0  # true point
        self.fp = 0  # false point
        self._name = name
    def update(self, preds, labels):
        sample_num = labels.shape[0] * labels.shape[2]  * labels.shape[3]
        preds = np.argmax(preds, axis=1)
        preds = np.expand_dims(preds, axis=1)
        right = (preds == labels).sum()
        self.tp += right
        self.fp += sample_num - right
        return right/sample_num  #设置

    def reset(self): #每一次计算后重新置零
        self.tp = 0
        self.fp = 0
    def accumulate(self):
        ap=self.fp+self.tp
        return float(self.tp) / ap if ap != 0 else .0
    def name(self):
        return self._name

#模型训练
model = paddle.Model(model)
model.prepare(
    paddle.optimizer.Adam(parameters=model.parameters()),
    paddle.nn.CrossEntropyLoss(axis=1),
    Selfacc()
)

#可视化训练过程
#绘制“正确率”变化图
import paddle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as font_manager
my_font = font_manager.FontProperties(fname='/usr/share/fonts/fangzheng/FZSYJW.TTF', size=16)

train_loader = paddle.io.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
model = paddle.Model(selfunet(2))
model.prepare(
    optimizer = paddle.optimizer.Adam(parameters=model.parameters()),
    loss = paddle.nn.CrossEntropyLoss(axis=1),
    metrics = Selfacc()
    )
#设置损失和精度确认方法
epoch_num = EPOCHS
costs = []
accs = []
for epoch in range(epoch_num):
    for batch_id, batch_data in enumerate(train_loader):
        inputs = batch_data[0]
        labels = batch_data[1]
        out = model.train_batch([inputs], [labels])
        costs.append(out[0])
        accs.append(out[1])
        print(out)
        if batch_id%100 == 0:
            print('epoch: {}, batch:{}, loss:{}, acc:{}'.format(epoch, batch_id, out[0], out[1]))


costs = np.array(costs)
acc = np.array(accs)

plt.figure()
plt.title('loss/损失值',fontproperties=my_font)
plt.plot(costs[:,0,0], color="orange")
plt.figure()
plt.title('acc/正确率',fontproperties=my_font)
plt.plot(acc[:,0], color="orange")
plt.show()

#模型评估与预测
# 评估模型
eval_result = model.evaluate(test_dataset, verbose = 1)
# 使用模型预测
predict_data = model.predict(test_dataset)

#结果可视化
import matplotlib.pyplot as plt
plt.figure()
i = 0
mask_idx = 0
plt.rcParams.update({"font.size":100})

plt.figure(figsize=(64, 48))
idxs = [3, 6,256]
i = 0
for idx in idxs:
    img, label = paddle.transpose(test_dataset[idx][0],(1,2,0)), test_dataset[idx][1][0,:,:]
    result_pre = predict_data[0][idx][0]
    result = np.argmax(result_pre, axis=0)

    plt.subplot(len(idxs), 3, i + 1)
    plt.imshow(img)
    plt.title('Input Image',fontsize=100)
    plt.axis("off")
    plt.subplot(len(idxs), 3, i + 2)
    plt.imshow(label, cmap = 'gray')
    plt.title('Label',fontsize=100)
    plt.axis("off")
    plt.subplot(len(idxs), 3, i + 3)
    plt.imshow(result.astype('uint8'), cmap = 'gray')
    plt.title('Predicted',fontsize=100)
    plt.axis("off")

    i = i + 3
plt.show()
