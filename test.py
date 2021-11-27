import numpy as np

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

def Calculate_total_acc_ppv_sen_spec(matrix):
    tn=matrix[0][0]
    fp=np.sum(matrix,axis=1)[0]-tn
    fn=np.sum(matrix,axis=0)[0]-tn
    tp=np.sum(matrix)-tn-fp-fn
    tp=12014
    tn=87346
    fp=1982
    fn=3695
    print(tp+tn+fp+fn)
    acc = (tp + tn) / (tp + tn + fp + fn)
    ppv = tp / (tp + fp)
    sen = tp / (tp + fn)
    spec = tn / (tn + fp)
    print('tp:{0},tn:{1},fp:{2},fn:{3}'.format(tp,tn,fp,fn))
    print('total：acc:{0:.2f}%,ppv:{1:.2f}%,sen:{2:.2f}%,spec:{3:.2f}%'.format(acc*100,ppv*100,sen*100,spec*100))

cm=[
    [39998,1692,827,1715,9],
    [619,319,797,1,1],
    [402,77,2687,54,0],
    [203,8,30,147,0],
    [3,1,3,0,0]
]

cm2=[
[88507	,389	,761	,663	,11],
[1174	,1073	,530	,1	,3],
[158	,526	,6369	,71	,5],
[114	,2	,59	,626	,1],
[1013	,1	,5	,0	,2875]
]

# Calculate_acc_ppv_sen_spec(np.asarray(cm),class_num=5,class_names=['N','S','V','F','Q'])
Calculate_total_acc_ppv_sen_spec(cm2)

a=[77.07,96.28,96.43,96.40,99.98]
print(sum(a)/5)


