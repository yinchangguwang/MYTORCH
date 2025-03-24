import mytorch.nn as nn
import mytorch.optim as optim
from mytorch.nn import Tensor
import mytorch.utils as utils
import numpy as np

# 从文件中导入数据
X = np.load('X.npy')
y = np.load('y.npy')

# 将数据转换为Tensor
X = Tensor(X, requires_grad=True)
y = Tensor(y, requires_grad=True)


# 划分数据集
X_train, X_test, y_train, y_test = utils.train_test_split(X, y, test_size=0.2)


# 构建一个简单的神经网络
model = nn.Sequential(
    nn.Linear(2, 10),
    nn.ReLU(),
    nn.Linear(10, 1),
    nn.Sigmoid()
)

# 初始化损失函数和优化器
criterion = nn.MSELoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 设置批量大小
batch_size = 32

# 训练模型
for epoch in range(100):    
    # 分批训练
    for i in range(0, X_train.data.shape[0], batch_size):
        end = min(i + batch_size, X_train.data.shape[0])
        X_batch = Tensor(X_train.data[i:end], requires_grad=True)
        y_batch = Tensor(y_train.data[i:end], requires_grad=True)

        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()
    
    # 计算训练精度
    train_predictions = model(X_train)
    train_accuracy = utils.compute_accuracy(train_predictions, y_train)
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}, Loss: {loss.data}, Train Accuracy: {train_accuracy * 100:.2f}%')
        

# 测试模型
predictions = model(X_test)
test_loss = criterion(predictions, y_test)
test_accuracy = utils.compute_accuracy(predictions, y_test)
print(f'Test Loss: {test_loss.data}, Test Accuracy: {test_accuracy * 100:.2f}%')