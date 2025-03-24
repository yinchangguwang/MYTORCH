import numpy as np
from .Tensor import Tensor

class Module:
    def __init__(self):
        self._parameters = {}
        self._modules = {}

    def parameters(self):
        return [p for p in self._parameters.values()]

    def add_module(self, name, module):
        self._modules[name] = module
        
    def add_parameter(self, name, param):
        self._parameters[name] = param
        super().__setattr__(name, param)

    def __setattr__(self, name, value):
        if isinstance(value, Module):
            self.add_module(name, value)
        elif isinstance(value, Tensor):
            self.add_parameter(name, value)
        else:
            super().__setattr__(name, value)
            

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        
        # 重置随机种子以生成随机的初始权重
        np.random.seed(None)
        
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Tensor(np.random.randn(out_features, in_features) * np.sqrt(2. / in_features), requires_grad=True)
        if bias:
            self.bias = Tensor(np.random.uniform(-1, 0, size=(out_features,)), requires_grad=True)
        else:
            self.bias = None
    
    def forward(self, x):
        if self.bias is None:
            return x.matmul(self.weight.T)
        return x.matmul(self.weight.T) + self.bias
    
    def __call__(self, x):
        return self.forward(x)
    
class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        
        # 重置随机种子以生成随机的初始权重
        np.random.seed(None)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = Tensor(np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2. / (in_channels * kernel_size * kernel_size)), requires_grad=True)
        self.bias = Tensor(np.zeros(out_channels), requires_grad=True)

    def forward(self, x):
        batch_size, in_channels, height, width = x.data.shape
        out_height = (height - self.kernel_size + 2 * self.padding) // self.stride + 1
        out_width = (width - self.kernel_size + 2 * self.padding) // self.stride + 1
        out = np.zeros((batch_size, self.out_channels, out_height, out_width))

        x_padded = np.pad(x.data, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        for i in range(out_height):
            for j in range(out_width):
                x_slice = x_padded[:, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size]
                for k in range(self.out_channels):
                    out[:, k, i, j] = np.sum(x_slice * self.weight.data[k, :, :, :], axis=(1, 2, 3))

        out += self.bias.data[None, :, None, None]
        return Tensor(out, requires_grad=self.requires_grad)

    def __call__(self, x):
        return self.forward(x)    
    
class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, x):
        batch_size, in_channels, height, width = x.data.shape
        out_height = (height - self.kernel_size + 2 * self.padding) // self.stride + 1
        out_width = (width - self.kernel_size + 2 * self.padding) // self.stride + 1
        out = np.zeros((batch_size, in_channels, out_height, out_width))

        x_padded = np.pad(x.data, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        for i in range(out_height):
            for j in range(out_width):
                x_slice = x_padded[:, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size]
                out[:, :, i, j] = np.max(x_slice, axis=(2, 3))

        return Tensor(out, requires_grad=self.requires_grad)

    def __call__(self, x):
        return self.forward(x)

class Sequential(Module):
    def __init__(self, *args):
        super().__init__()
        self.layers = args
        # 自动注册子层
        for idx, layer in enumerate(args):
            self.add_module(f"layer_{idx}", layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __call__(self, x):
        return self.forward(x)
    
    def parameters(self):
        # 递归收集所有子层参数
        params = []
        for name, module in self._modules.items():
            params.extend(module.parameters())
        return params
    
# 激活函数
class ReLU(Module):
    def forward(self, x):
        return x.relu()

    def __call__(self, x):
        return self.forward(x)
    
class Sigmoid(Module):
    def forward(self, x):
        return x.sigmoid()
    
    def __call__(self, x):
        return self.forward(x)
    
class Tanh(Module):
    def forward(self, x):
        return x.tanh()
    
    def __call__(self, x):
        return self.forward(x)
    
# class Softmax(Module):
#     def forward(self, x):
#         return x.softmax()
    
#     def __call__(self, x):
#         return self.forward(x)

    
# 损失函数
class MSELoss(Module):    
    def forward(self, prediction, target):
        loss = ((prediction - target) ** 2).mean() * 0.5
        # print(loss.data)
        loss._prev = [prediction]
        loss._op = 'mse'
        loss.grad = prediction.data - target.data
        # print(loss.grad)
        return loss
    
    def __call__(self, pred, target):
        return self.forward(pred, target)
            
class CrossEntropyLoss(Module):
    def forward(self, prediction, target):
        # 使用softmax计算概率分布
        probs = prediction.softmax()
        # 计算交叉熵损失
        loss = -np.sum(target.data * np.log(probs.data + 1e-9)) / target.data.shape[0]
        Loss = Tensor(loss, requires_grad=True)
        Loss._prev = [prediction]
        Loss._op = 'cross_entropy'
        loss.grad = prediction.data - target.data
        return Loss

    def __call__(self, prediction, target):
        return self.forward(prediction, target)
    

    

    