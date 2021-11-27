#The training is conducted in Chazal's data division mode
import wfdb
import pywt
import argparse
import numpy as np
import os
from torch.utils import data
import torch
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import Model
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from matplotlib.pyplot import MultipleLocator
import matplotlib.pyplot as plt
from focalloss import FocalLoss

class MyDataset(data.Dataset):
    def __init__(self,datas,labels):
        self.Data = np.asarray(datas)  # 数据集
        self.Label = np.asarray(labels)  # 这是数据集对应的标签

    def __getitem__(self, index):
        # 把numpy转换为Tensor
        data = torch.from_numpy(self.Data[index]).reshape(1,16,16)
        label = torch.tensor(self.Label[index])
        return data, label

    def __len__(self):
        return len(self.Data)

#计算acc,ppv,sen,spec,all_acc
def Calculate_acc_ppv_sen_spec(matrix,class_num,class_names):
    results_matrix = np.zeros([class_num,4])
    #diagonal负责统计对角线元素的和
    diagonal=0
    for i in range(class_num):
        tp = matrix[i][i]
        diagonal+=tp
        fn = np.sum(matrix,axis=1)[i] - tp
        fp=np.sum(matrix,axis=0)[i]-tp
        tn=np.sum(matrix)-tp-fp-fn
        acc=(tp+tn)/(tp+tn+fp+fn)
        ppv=tp/(tp+fp)
        sen=tp/(tp+fn)
        spec=tn/(tn+fp)
        results_matrix[i][0]=acc*100
        results_matrix[i][1] = ppv * 100
        results_matrix[i][2] = sen * 100
        results_matrix[i][3] = spec * 100
    for i in range(class_num):
        print('{0}：acc:{1:.2f}%,ppv:{2:.2f}%,sen:{3:.2f}%,spec:{4:.2f}%'.format(class_names[i],results_matrix[i][0],
                                                                     results_matrix[i][1],results_matrix[i][2],results_matrix[i][3]))
    print('模型总体精度acc为：{0:.2f}%'.format((diagonal/matrix.sum())*100))

def S_shaped(x):
    x=x.tolist()
    for i in range(1, 16, 2):
        x[i * 16:i * 16 + 16].reverse()
    return x

def Denoising(file_number):
    #用db6作为小波基，对心电数据进行9尺度小波变换
    record = wfdb.rdrecord(os.path.join('./','MIT-Data','score-data',file_number), channel_names=['MLII'])
    signal = record.p_signal.flatten()
    coeffs = pywt.wavedec(data=signal, wavelet='db6', level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    # 将高频信号cD1、cD2、cD3置零
    cD1.fill(0)
    cD2.fill(0)
    cD3.fill(0)
    #去基漂
    cA9.fill(0)
    # 将其他中低频信号按软阈值公式滤波
    for i in range(1, len(coeffs) - 3):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)
    rdata = pywt.waverec(coeffs=coeffs, wavelet='db6')

    return rdata

# 读取心电数据和对应标签,并对数据进行小波去噪
def GetData(file_number, X_data, Y_data):
    # 五大类里面的小分支
    N = ['N', 'L', 'R', 'e', 'j']
    S = ['A', 'a', 'J', 'S']
    V = ['V', 'E']
    F = ['F']
    Q = ['/', 'f', 'Q']
    # 五大类
    Classset = ['N', 'S', 'V', 'F', 'Q']
    EcgClassSet = {'N': N, 'S': S, 'V': V, 'F': F, 'Q': Q}

    # 读取心电数据记录
    print("正在读取 " + file_number+ " 号心电数据")
    # 小波去噪
    rdata = Denoising(file_number)
    # 获取心电数据记录中R波的位置和对应的标签
    annotation=annotation = wfdb.rdann(os.path.join('./','MIT-Data','score-data',file_number), 'atr')
    Rlocation = annotation.sample
    Rclass = annotation.symbol
    rdata_length=len(rdata)
    # 因为只选择NAVLR五种心电类型,所以要选出该条记录中所需要的那些带有特定标签的数据,舍弃其余标签的点
    # X_data在R波前后截取长度为256的数据点
    # Y_data将NAVLR按顺序转换为01234
    for i in range(len(annotation.symbol)) :
            if Rlocation[i]<128 or Rlocation[i]>rdata_length-128:
                continue
            else :
                Subcategory.add(Rclass[i])
                key = ''
                for item in EcgClassSet.items():
                    if Rclass[i] in item[1]:
                        key = item[0]
                if key == '':
                    continue
                lable = Classset.index(key)
                x_train = rdata[Rlocation[i] - 128:Rlocation[i] + 128]

                #z分数归一化
                mu = x_train.mean()
                std = x_train.std()
                for i in range(len(x_train)):
                    x_train[i] = (x_train[i] - mu) / std

                X_data.append(x_train)
                Y_data.append(lable)

def Dividing_datasets_into_DS1andDS2():
    DS1_numbers = [101,106,108,109,112,114,115,116,
                   118,119,122,124,201,203,205,207,
                   208,209,215,220,223,230]

    DS2_numbers = [100,103,105,111,113,117,121,123,
                   200,202,210,212,213,214,219,221,
                   222,228,231,232,233,234]

    DS1_dataSet = []
    DS1_labelSet = []
    DS2_dataSet = []
    DS2_labelSet = []

    for file_number in DS1_numbers:
        GetData(str(file_number), DS1_dataSet, DS1_labelSet)
    print('总共数据小类别：', Subcategory)
    Subcategory.clear()

    for file_number in DS2_numbers:
        GetData(str(file_number), DS2_dataSet, DS2_labelSet)
    print('总共数据小类别：', Subcategory)

    DS1_dataSet = np.array(DS1_dataSet).reshape(-1, 256)
    DS1_lableSet = np.array(DS1_labelSet).reshape(-1, 1)
    DS2_dataSet = np.array(DS2_dataSet).reshape(-1, 256)
    DS2_lableSet = np.array(DS2_labelSet).reshape(-1, 1)

    train_ds = np.hstack((DS1_dataSet, DS1_lableSet))
    np.random.shuffle(train_ds)
    DS1_X = train_ds[:, :256].reshape(-1, 256)
    DS1_Y = train_ds[:, 256]

    train_ds = np.hstack((DS2_dataSet, DS2_lableSet))
    np.random.shuffle(train_ds)
    DS2_X = train_ds[:, :256].reshape(-1, 256)
    DS2_Y = train_ds[:, 256]

    return DS1_X,DS1_Y,DS2_X,DS2_Y

def train():
    # 训练
    train_dataset = MyDataset(train_x, train_y)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=0)

    net = Model.Get_se_resnet_19_2d(num_classes=5).cuda()

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = FocalLoss(class_num=5)

    net.train()

    for epoch in range(args.epochs):
        running_loss = 0
        i = 0
        batch = tqdm(train_dataloader)
        for inputs, labels in batch:
            inputs, labels = inputs.type(torch.FloatTensor).cuda(), labels.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.item()
            i += 1
            batch.set_description('第{0}轮训练，当前损失平均值为：{1:.10f}'.format(epoch + 1, running_loss / i))
        scheduler.step()

    return net

def test(net):
    test_dataset = MyDataset(test_x, test_y)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,num_workers=0)

    net.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        batch = tqdm(test_dataloader)
        for inputs, labels in batch:
            inputs, labels = inputs.type(torch.FloatTensor).cuda(), labels.cuda()
            outputs = net(inputs)
            _, predicted = torch.max(outputs, dim=1)

            for i in range(len(labels)):
                label = labels[i]
                y_true.append(label.item())
                y_pred.append(predicted[i].item())

    print('测试集acc为：{0:.3f}%'.format(accuracy_score(y_true, y_pred) * 100))

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    return cm

if __name__=='__main__':

    Subcategory = set()

    DS1_X,DS1_Y,DS2_X,DS2_Y=Dividing_datasets_into_DS1andDS2()

    parser = argparse.ArgumentParser(description='train_and_test')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='epochs')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    args = parser.parse_args()

    print('训练集的总样本数为：{0}'.format(len(DS1_X)))

    # 数据再处理
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    for x in DS1_X:
        x = S_shaped(x)
        train_x.append(x)
    for y in DS1_Y:
        train_y.append(int(y))
    for x in DS2_X:
        x = S_shaped(x)
        test_x.append(x)
    for y in DS2_Y:
        test_y.append(int(y))

    print('训练集中，N：{0}，S：{1}，V：{2}，F：{3}，Q：{4}'.format(train_y.count(0), train_y.count(1), train_y.count(2),
                                                      train_y.count(3), train_y.count(4)))

    net=train()

    cm=test(net)

    print(cm)

    Calculate_acc_ppv_sen_spec(cm, class_num=5, class_names=['N', 'S', 'V', 'F', 'Q'])


