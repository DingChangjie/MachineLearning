import numpy as np

def calc_entropy(data):
    """
    本函数根据输入数据集，计算它的信息熵
    data : 输入数据集

    """
    # 统计类值各自出现的频率
    class_num = np.unique(data[:,-1],False,False,True)[1]
    # 计算当前数据集的信息熵
    class_num_uni = class_num/np.sum(class_num) # 归一化
    class_num_log = np.log2(class_num_uni)
    I = np.dot(-class_num_uni,class_num_log)
    return I,class_num
    
def split_data(data,attribute):
    """
    本函数将指定矩阵data自动按属性attribute切分成子矩阵，并返回这些子矩阵
    data : 输入数据集
    attribute : 选择的属性

    """
    # 获得该属性的值表，即那些i
    unique_value = np.unique(data[:,attribute],False,False,True)[0]
    # 结果将作为列表返回
    ret = []
    # 根据当前属性的值表切分矩阵
    for value in unique_value:
        index = np.argwhere(data[:,attribute] == value)
        index = index.ravel() # 降掉多出来的一维
        tmp = data[index,:]
        tmp = np.delete(tmp,attribute,axis=1) # 所有子矩阵删除该属性列(并不影响calc_entropy的统计结果)
        ret.append(tmp) # 最终返回的是去掉该属性的子矩阵
    return ret

def make_choice(data):
    """
    本函数用于决断该选择哪个属性
    data : 输入数据集

    """
    num_attri = data.shape[1]-1 # 统计有多少个属性
    # 求基本熵I(p,n)和p,n
    base_entropy,pn = calc_entropy(data)
    # 存储各个属性的信息增益
    G = np.zeros(num_attri)
    # 对每个属性，都求它的信息熵
    for j in range(0,num_attri):
        sum_nume = 0 # 用于求分子
        for subset in split_data(data,j):
            I,class_num=calc_entropy(subset)
            sum_nume = np.sum(class_num)*I + sum_nume
        # 求该属性的总信息熵
        E = sum_nume/np.sum(pn)
        # 录入属性j的信息增益
        G[j] = base_entropy-E
    # 作出选择
    return(np.argmax(G))

def vote(data):
    """
    本函数用多数投票机制，解决属性不够用的情况下，样本的分类问题
    这时哪一类占比最大，决策树就判定样本可能属于哪一类
    data : 输入数据集（只剩最后那个class列了）

    """
    class_name = np.unique(data[:,-1],False,False,True)[0]
    class_num = np.unique(data[:,-1],False,False,True)[1]
    return int(class_name[np.argmax(class_num)])


def create_tree(data,attri_labels):
    """
    本函数将以递归的方式生成决策树
    data : 输入数据集
    attri_labels : 人工录入的属性标签

    """
    # 停止条件一：当前数据集data中，所有样本都分入了同一class
    if len(np.unique(data[:,-1],False,False,True)[0]) == 1 :
        return int(np.unique(data[:,-1],False,False,True)[0]) # 到达叶节点
    # 停止条件二：属性已用尽，样本还没分入同一class （属性没选够）
    elif data.shape[1] == 1 :
        return (vote(data)) # 到达叶节点

    # 求最佳属性，该属性对应的值表，最后贴上标签
    best_attri = make_choice(data)
    best_value = np.unique(data[:,best_attri],False,False,True)[0]
    num_best_value = len(best_value)
    label_best_attri = attri_labels[best_attri] # 最佳属性对应的标签
    # 利用字典搭建树的可视化结构
    tree = {label_best_attri:{}}
    del(attri_labels[best_attri]) # 与下面递归的split_data同步删除属性标签

    # 递归
    for value in range(0,num_best_value): # 遍历值表
        sub_labels = attri_labels[:]
        tree[label_best_attri][value] = create_tree((split_data(data,best_attri)[value]),sub_labels)
    
    # 返回值
    return tree
