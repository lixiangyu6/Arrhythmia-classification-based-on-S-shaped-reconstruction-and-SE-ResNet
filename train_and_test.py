import argparse
import numpy as np
import time
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

def S_shaped(x):
    x=x.tolist()
    for i in range(1, 16, 2):
        x[i * 16:i * 16 + 16].reverse()
    return x

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

#计算模型的总体acc,ppv,sen,spec
def Calculate_total_acc_ppv_sen_spec(matrix):
    tn=matrix[0][0]
    fp=np.sum(matrix,axis=1)[0]-tn
    fn=np.sum(matrix,axis=0)[0]-tn
    tp=np.sum(matrix)-tn-fp-fn
    acc = (tp + tn) / (tp + tn + fp + fn)
    ppv = tp / (tp + fp)
    sen = tp / (tp + fn)
    spec = tn / (tn + fp)
    print('tp:{0},tn:{1},fp:{2},fn:{3}'.format(tp,tn,fp,fn))
    print('total：acc:{0:.2f}%,ppv:{1:.2f}%,sen:{2:.2f}%,spec:{3:.2f}%'.format(acc*100,ppv*100,sen*100,spec*100))

#降成三个特征，但以二维形式表现
def PCA_show(features,labels):
    N_x,S_x,V_x,F_x,Q_x=[],[],[],[],[]
    N_y, S_y, V_y, F_y, Q_y = [], [], [], [], []
    total=[]
    for feature in features:
        feature = feature.cpu().numpy().reshape(1, -1)
        feature = feature.tolist()
        total.append(feature[0])
    pca = PCA(n_components=3)  # 加载PCA算法，设置降维后主成分数目
    reduced_x = pca.fit_transform(total)  # 对样本进行降维
    for i in range(len(labels)):
        if labels[i] == 0:
            N_x.append(reduced_x[i][0])
            N_y.append(reduced_x[i][1])
        elif labels[i] == 1:
            S_x.append(reduced_x[i][0])
            S_y.append(reduced_x[i][1])
        elif labels[i] == 2:
            V_x.append(reduced_x[i][0])
            V_y.append(reduced_x[i][1])
        elif labels[i] == 3:
            F_x.append(reduced_x[i][0])
            F_y.append(reduced_x[i][1])
        else:
            Q_x.append(reduced_x[i][0])
            Q_y.append(reduced_x[i][1])

    font = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 15,
             }
    # 设置刻度字体大小
    plt.tick_params(labelsize=15)
    plt.xlabel("d1",font)
    plt.ylabel("d2",font)
    # 设置坐标的显示范围
    plt.axis([-40, 40, -40, 40])
    # 设置刻度间隔
    ax = plt.gca()
    # x的刻度间隔
    plt.tick_params(labelsize=15)
    x_major_locator = MultipleLocator(20)
    y_major_locator = MultipleLocator(20)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.scatter(N_x, N_y, c='r', marker='x')
    plt.scatter(S_x, S_y, c='g', marker='x')
    plt.scatter(V_x, V_y, c='b', marker='x')
    plt.scatter(F_x, F_y, c='k', marker='x')
    plt.scatter(Q_x, Q_y, c='y', marker='x')
    plt.legend(['N', 'S', 'V', 'F', 'Q'],fontsize=15,loc='upper right')
    plt.grid()
    plt.show()

def PCA_SHOW(conv_out_list,labels):
    for i in range(len(conv_out_list)):
        conv_out=conv_out_list[i]
        conv_out.remove()
        PCA_show(conv_out.features, labels)

def train():
    # 训练
    train_dataset = MyDataset(train_x, train_y)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=0)

    net = Model.Get_se_resnet_19_2d(num_classes=5).cuda()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        net.train()
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
            batch.set_description('当前第{0}折，第{1}轮训练，当前损失平均值为：{2:.5f}'.format(num,epoch + 1, running_loss / i))

        # 验证
        valid_dataset = MyDataset(valid_x, valid_y)
        valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True,num_workers=0)

        net.eval()

        y_true = []
        y_pred = []

        with torch.no_grad():
            batch = tqdm(valid_dataloader)
            for inputs, labels in batch:
                inputs, labels = inputs.type(torch.FloatTensor).cuda(), labels.cuda()
                outputs = net(inputs)
                _, predicted = torch.max(outputs, dim=1)

                for i in range(len(labels)):
                    label = labels[i]
                    y_true.append(label.item())
                    y_pred.append(predicted[i].item())

        print('验证集acc为：{0:.3f}%'.format(accuracy_score(y_true, y_pred) * 100))
        acc.append(accuracy_score(y_true, y_pred) * 100)

        # 保存模型
        torch.save(net.state_dict(), os.path.join('./', 'model', 'net' + str(epoch) + '.pth'))

def test():
    # 测试
    max_acc = max(acc)
    max_acc_index = acc.index(max_acc)

    # 加载模型
    net = Model.Get_se_resnet_19_2d(num_classes=5).cuda()
    net.load_state_dict(torch.load(os.path.join('./', 'model', 'net' + str(max_acc_index) + '.pth')))

    test_dataset = MyDataset(test_x, test_y)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,num_workers=0)

    net.eval()

    # #PCA可视化
    # conv_out_list=[Model.LayerActivations(net.conv1,-1),Model.LayerActivations(net.layer1,-1),Model.LayerActivations(net.layer2,-1),
    #           Model.LayerActivations(net.layer3,-1),Model.LayerActivations(net.layer4,-1),Model.LayerActivations(net.avg_pool2d,-1)]

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

            # # PCA可视化
            # PCA_SHOW(conv_out_list,labels.cpu())

    print('测试集acc为：{0:.3f}%'.format(accuracy_score(y_true, y_pred) * 100))

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    cms.append(cm)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='train_and_test')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--epochs', type=int, default=30, help='epochs')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    args = parser.parse_args()

    # 开始计时
    start = time.perf_counter()

    all_x = np.load(os.path.join('./', 'data_save', 'AllData.npy'))
    all_y = np.load(os.path.join('./', 'data_save', 'AllLabel.npy'))

    #cms用来记录每一次的混淆矩阵
    cms = []

    num=0

    kf = KFold(n_splits=10)
    for train_valid_index, test_index in kf.split(all_x):
        train_valid_x_init, train_valid_y_init = all_x[train_valid_index], all_y[train_valid_index]
        test_x_init, test_y_init = all_x[test_index], all_y[test_index]

        #训练集按照9：1划分训练集与验证集
        train_x_init,valid_x_init, train_y_init,valid_y_init = train_test_split(train_valid_x_init, train_valid_y_init, test_size=0.1)

        smo = SMOTE()
        train_x_init_smote, train_y_init_smote = smo.fit_resample(train_x_init, train_y_init)
        print('对训练数据部分做smote平衡，平衡后总样本数为：{0}'.format(len(train_x_init_smote)))

        # 数据再处理
        train_x = []
        train_y = []
        valid_x = []
        valid_y = []
        test_x = []
        test_y = []
        for x in train_x_init_smote:
            x = S_shaped(x)
            train_x.append(x)
        for y in train_y_init_smote:
            train_y.append(int(y))
        for x in valid_x_init:
            x = S_shaped(x)
            valid_x.append(x)
        for y in valid_y_init:
            valid_y.append(int(y))
        for x in test_x_init:
            x = S_shaped(x)
            test_x.append(x)
        for y in test_y_init:
            test_y.append(int(y))
        print('训练集中，N：{0}，S：{1}，V：{2}，F：{3}，Q：{4}'.format(train_y.count(0),train_y.count(1),train_y.count(2),train_y.count(3),train_y.count(4)))

        #acc用来记录精确度以便挑选最好的模型
        acc=[]

        num+=1

        train()

        test()

    #展示结果
    cm=cms[0]
    for i in range(1,len(cms)):
        cm+=cms[i]

    Calculate_acc_ppv_sen_spec(cm,class_num=5,class_names=['N','S','V','F','Q'])
    Calculate_total_acc_ppv_sen_spec(cm)

    print('共耗时{0}秒'.format(time.perf_counter()-start))








