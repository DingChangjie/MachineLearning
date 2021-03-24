# 这里是主程序
import numpy as np
import Training
from Processing import save_tree, load_tree
from Predicting import predict

if __name__ == "__main__":
    # 1. 训练
    data = np.loadtxt('Sample_Training.txt',dtype=int)
    attri_labels = ['Outlook','Temperature','Humidity','Windy']
    my_tree = Training.create_tree(data,attri_labels)
    # 保存训练好的决策树
    save_tree(my_tree,'ID3')

    # 2. 预测
    # 读取决策树，并进行预测
    loaded_tree = load_tree('ID3')
    attri_labels = ['Outlook','Temperature','Humidity','Windy']
    data_test = np.loadtxt('Sample_test.txt',dtype=int)
    data_test = data_test - 1 # 调试：样本属性序号必须从0开始
    data_ref = np.loadtxt('Sample_reference.txt',dtype=int)
    predict(data_test,attri_labels,loaded_tree,calc_accu=data_ref)

