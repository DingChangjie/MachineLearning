# 本文件使用一个创建好的决策树进行预测, 支持矩阵输入，可以输出准确率
import numpy as np
import copy

def _concatenate(data,attri_labels):
    """
    [保护]用于将data和attri_labels重新拼合起来(重映射)，只能处理单行数据
    data : 指data的某一行，且没有class列
    attri_labels : 人工录入的属性标签

    """
    num_attri = len(data)
    if not num_attri == len(attri_labels):
        print ("ERROR : 标签数和数据集的属性数不一致，请检查输入的是不是待预测集")
        return -1
    else:
        concatenated = {}
        for j in range(0,num_attri):
            concatenated[attri_labels[j]]=data[j]
        return concatenated

def _predict_single(data,tree):
    """
    [保护]本函数根据输入数据集和决策树，来预测输入数据集的分类
    data : 输入数据集(不含class的)
    tree : 决策树，为嵌套字典

    """
    # 正常停止条件：达到叶节点
    if not isinstance(tree,dict):
        return tree
    
    # 遍历搜索待预测数据集的属性
    get_key = "0"
    for key in data.keys():
        if key in tree.keys():
            get_key = key
            break
    
    # 决策树缩编，露出来的子字典就是该属性的值表
    sub_tree = tree.pop(get_key)
    # 此时，data[key]的值应该与子字典中的某个key对应。.get方法恰好能返回这个子字典
    sub_tree = sub_tree.get(data[get_key])
    data.pop(get_key) # 数据集缩编
    result = _predict_single(data,sub_tree) # 递归
    # 由第一层递归返回值
    return result

def _calc_accuracy(predicted,reference):
    """
    [保护]本函数用于计算预测结果和参考数据之间的准确度

    """
    predicted = np.array(predicted)
    compared = (predicted == reference['calc_accu']) + 0 # 关键字参数的返回值是字典
    return (np.sum(compared)/len(predicted))


def predict(dataset,attri_labels,tree,**calc_accu):
    """
    本函数是predict模块的接口，支持numpy数组功能
    dataset : 数组形式的输入的待分类数据
    attri_labels : 人工录入的属性标签
    tree : 决策树
    calc_accu : [Keyword Argument]计算准确率，本参数要求以list给出参考数据
    ===================
    TODO : 默认输入是一个矩阵，若只有一行，将发生错误
    TODO : 没有加入查错（检查属性标签数和输入数据列数是否相等）功能

    """
    predicted = []
    num_obj = dataset.shape[0]
    tree_copied = copy.deepcopy(tree) # 必须使用深拷贝的方法，保留一份tree的副本
    for i in range(0,num_obj):
        i_concatenated = _concatenate(dataset[i,:],attri_labels)
        tree_using = copy.deepcopy(tree_copied) # 每次随用随取
        result = _predict_single(i_concatenated,tree_using)
        print("样本{0}属于类别{1}".format(i,result))
        predicted.append(result)
        del tree_using # 用完删除，释放空间
    
    # 可选项：计算准确率，默认保留三位小数
    if calc_accu :
        accuracy = _calc_accuracy(predicted,calc_accu)
        print ("预测准确率为：{0:.3%}".format(accuracy))

# 调试_concatenate模块
# loaded_tree = load_tree('ID3')
# attri_labels = ['Outlook','Temperature','Humidity','Windy']
# data_test = np.loadtxt('Sample_test_3.txt',dtype=int)
# i_concatenated = _concatenate(data_test,attri_labels)
# print(_predict_single(i_concatenated,loaded_tree))    