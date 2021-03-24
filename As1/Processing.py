# 本模块提供一些数据处理的小工具
import pickle
import numpy as np

# 第一部分：通用工具
# 决策树工具
def save_tree(tree,name):
    """
    本函数将生成的决策树保存为pickle格式
    tree:生成的决策树
    name:要保存的文件名

    """
    with open(name,'wb') as f_in:
        pickle.dump(tree,f_in)

def load_tree(name):
    """
    本函数用于打开已保存的决策树

    """
    with open(name,'rb') as f_out:
        loaded = pickle.load(f_out)
    return loaded

# 第二部分：DNA决策树训练的专用工具
# 1. 待测DNA数据集的处理
def dna_generate_labels():
    """
    本函数自动读取DNA/dna.data文件内的数据，并根据数据量生成标签

    """
    data = np.loadtxt('DNA/dna.data',dtype=int)
    data_num = data.shape[1] - 1 # 除了最后一列外，都是属性
    ret = ["A{0}".format(i) for i in range(0,data_num)]
    return ret
