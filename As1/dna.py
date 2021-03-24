import numpy as np
import Training
from Processing import save_tree, load_tree, dna_generate_labels
from Predicting import predict

if __name__ == "__main__":
    # 1. 训练
    # data = np.loadtxt('./DNA/dna.data',dtype=int)
    # attri_labels = dna_generate_labels()
    # #print(attri_labels)
    # #print(type(attri_labels))
    # my_tree = Training.create_tree(data,attri_labels)
    # # 保存训练好的决策树
    # save_tree(my_tree,'ID3')

    # 2. 预测
    loaded_tree = load_tree('ID3')
    attri_labels =dna_generate_labels()
    data_test = np.loadtxt('DNA/dna_test.txt',dtype=int)
    data_ref = np.loadtxt('DNA/dna_reference.txt',dtype=int)
    predict(data_test,attri_labels,loaded_tree,calc_accu=data_ref)