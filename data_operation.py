import wfdb
import pywt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import os

def Denoising(file_number):
    #用db6作为小波基，对心电数据进行9尺度小波变换
    record = wfdb.rdrecord(os.path.join('./','MIT-Data','score-data',file_number), channel_names=['MLII'])
    signal = record.p_signal.flatten()

    # # 坐标名称的字体设置
    # font = {'family': 'Times New Roman',
    #          'weight': 'normal',
    #          'size': 20,
    #          }
    # plt.figure(figsize=(20, 4))
    # plt.plot(signal[0:1500],'r')
    # #对图像的边框进行修改
    # ax = plt.axes()
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # plt.tick_params(labelsize=15)
    # plt.xlabel("Length",font)
    # plt.ylabel("Voltage/mV",font)
    # # 设置坐标的显示范围
    # plt.axis([0, 1500, -0.8, 1.2])
    # # 设置刻度间隔
    # # x的刻度间隔
    # x_major_locator = MultipleLocator(300)
    # # y的刻度间隔
    # y_major_locator = MultipleLocator(0.2)
    # ax = plt.gca()
    # ax.xaxis.set_major_locator(x_major_locator)
    # ax.yaxis.set_major_locator(y_major_locator)
    # plt.legend(['no denoising'],fontsize=20)
    # plt.show()

    coeffs = pywt.wavedec(data=signal, wavelet='db6', level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    # 将高频信号cD1、cD2置零
    cD1.fill(0)
    cD2.fill(0)
    #去基漂
    cA9.fill(0)
    # 将其他中低频信号按软阈值公式滤波
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)
    rdata = pywt.waverec(coeffs=coeffs, wavelet='db6')

    # plt.figure(figsize=(20, 4))
    # plt.plot(rdata[0:1500],'r')
    # ax = plt.axes()
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # plt.tick_params(labelsize=15)
    # plt.xlabel("Length",font)
    # plt.ylabel("Voltage/mV",font)
    # # 设置坐标的显示范围
    # plt.axis([0, 1500, -0.8, 1.2])
    # # 设置刻度间隔
    # # x的刻度间隔
    # x_major_locator = MultipleLocator(300)
    # # y的刻度间隔
    # y_major_locator = MultipleLocator(0.2)
    # ax = plt.gca()
    # ax.xaxis.set_major_locator(x_major_locator)
    # ax.yaxis.set_major_locator(y_major_locator)
    # plt.legend(['denoising'],fontsize=20)
    # plt.show()

    return rdata

# 读取心电数据和对应标签,并对数据进行小波去噪
def GetData(file_number, X_data, Y_data):
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

if __name__=='__main__':

    Subcategory = set()

    # 五大类里面的小分支
    N = ['N', 'L', 'R', 'e', 'j']
    S = ['A', 'a', 'J', 'S']
    V = ['V', 'E']
    F = ['F']
    Q = ['/', 'f', 'Q']
    # 五大类
    Classset = ['N', 'S', 'V', 'F', 'Q']
    EcgClassSet = {'N': N, 'S': S, 'V': V, 'F': F, 'Q': Q}

    file_numbers = [100, 101, 103, 105, 106, 107,108, 109, 111, 112,113, 114, 115, 116,
                    117, 118, 119, 121, 122,123, 124, 200, 201, 202, 203, 205, 207, 208,
                    209, 210, 212, 213, 214, 215,217, 219, 220, 221,222, 223, 228, 230,
                    231, 232, 233, 234]

    dataSet=[]
    labelSet=[]

    for file_number in file_numbers:
        GetData(str(file_number), dataSet, labelSet)
    dataSet = np.array(dataSet).reshape(-1, 256)
    lableSet = np.array(labelSet).reshape(-1, 1)

    #展示心拍片段
    Heart_beat_display={'N':[], 'S':[],'V':[], 'F':[], 'Q':[]}
    for i in range(len(dataSet)):
        data=dataSet[i]
        label=Classset[labelSet[i]]
        if len(Heart_beat_display[label])==0:
            Heart_beat_display[label].append(data)
            print(label)
            ax = plt.axes()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.xlabel("Length")
            plt.ylabel("Voltage/mV")
            plt.axis([-5, 256, -5, 8])
            plt.plot(data)
            plt.show()

    #横向堆叠
    train_ds = np.hstack((dataSet, lableSet))
    #转numpy数组,打乱顺序
    np.random.shuffle(train_ds)
    # 数据集及其标签集
    X = train_ds[:, :256].reshape(-1, 256)
    Y = train_ds[:, 256]

    #divide按照训练与测试划分数据，all则不划分
    way='all'

    if way=='divide':
        # 测试集及其标签集
        RATIO = 0.1
        shuffle_index = np.random.permutation(len(X))
        test_length = int(RATIO * len(shuffle_index))  # RATIO = 0.1
        #按照9:1划分总体数据
        test_index = shuffle_index[:test_length]
        train_index = shuffle_index[test_length:]
        X_train, Y_train = X[train_index], Y[train_index]
        Y_train = Y_train.reshape(1, -1)
        X_test, Y_test = X[test_index], Y[test_index]
        Y_test = Y_test.reshape(1, -1)
        np.save(os.path.join('./', 'data_save', 'TrainData.npy'), X_train)
        np.save(os.path.join('./', 'data_save', 'TrainLabel.npy'), Y_train)
        np.save(os.path.join('./', 'data_save', 'TestData.npy'), X_test)
        np.save(os.path.join('./', 'data_save', 'TestLabel.npy'), Y_test)
        print('训练集数据和测试集数据保存完毕')
    elif way=='all':
        np.save(os.path.join('./', 'data_save', 'AllData.npy'), X)
        np.save(os.path.join('./', 'data_save', 'AllLabel.npy'), Y)
        print('全部数据保存完毕')

    print('总共数据小类别：', Subcategory)
    print('数据总数', len(X))






