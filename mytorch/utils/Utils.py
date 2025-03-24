from mytorch.nn import Tensor
import numpy as np

def train_test_split(X, y, test_size=0.2, random_seed=42):
    # 设置随机种子
    np.random.seed(random_seed)

    # 获取数据集大小
    num_samples = X.data.shape[0]
    
    # 计算训练集和测试集大小
    num_train = int((1 - test_size) * num_samples)
    indices = np.arange(num_samples)
    
    # 打乱数据
    np.random.shuffle(indices)
    
    # 划分训练集和测试集
    train_indices = indices[:num_train]
    test_indices = indices[num_train:]

    # 返回训练集和测试集数据
    X_train = Tensor(X.data[train_indices], requires_grad=True)
    y_train = Tensor(y.data[train_indices], requires_grad=True)
    X_test = Tensor(X.data[test_indices], requires_grad=True)
    y_test = Tensor(y.data[test_indices], requires_grad=True)

    return X_train, X_test, y_train, y_test
