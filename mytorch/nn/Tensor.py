import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._prev = [] # 保存操作链
        self._op = None # 记录是什么操作生成的
        self._power = None # 记录幂次
        
    def op(self):
        return self._op

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"
    
    # 支持广播机制
    def _broadcast(self, other):
        if isinstance(other, Tensor):
            return np.broadcast_to(other.data, self.data.shape)
        return other

    # 基本运算支持
    def __add__(self, other):
        other_data = self._broadcast(other)
        out = Tensor(self.data + other_data, requires_grad=self.requires_grad or (isinstance(other, Tensor) and other.requires_grad))
        out._prev = [self, other] if isinstance(other, Tensor) else [self]
        out._op = 'add'
        return out

    def __sub__(self, other):
        other_data = self._broadcast(other)
        out = Tensor(self.data - other_data, requires_grad=self.requires_grad or (isinstance(other, Tensor) and other.requires_grad))
        out._prev = [self, other] if isinstance(other, Tensor) else [self]
        out._op = 'sub'
        return out

    def __mul__(self, other):
        other_data = self._broadcast(other)
        out = Tensor(self.data * other_data, requires_grad=self.requires_grad or (isinstance(other, Tensor) and other.requires_grad))
        out._prev = [self, other] if isinstance(other, Tensor) else [self]
        out._op = 'mul'
        return out

    def __truediv__(self, other):
        other_data = self._broadcast(other)
        out = Tensor(self.data / other_data, requires_grad=self.requires_grad or (isinstance(other, Tensor) and other.requires_grad))
        out._prev = [self, other] if isinstance(other, Tensor) else [self]
        out._op = 'div'
        return out

    def __pow__(self, power):
        out = Tensor(self.data ** power, requires_grad=self.requires_grad)
        out._prev = [self]
        out._op = 'pow'
        out._power = power
        return out

    def __neg__(self):
        out = Tensor(-self.data, requires_grad=self.requires_grad)
        out._prev = [self]
        out._op = 'neg'
        return out

    # 矩阵运算
    def matmul(self, other):
        out = Tensor(np.dot(self.data, other.data), requires_grad=self.requires_grad or other.requires_grad)
        out._prev = [self, other]
        out._op = 'matmul'
        return out

    # 支持索引切片
    def __getitem__(self, index):
        return Tensor(self.data[index], requires_grad=self.requires_grad)

    # 激活函数
    def relu(self):
        out = Tensor(np.maximum(0, self.data), requires_grad=self.requires_grad)
        out._prev = [self]
        out._op = 'relu'
        return out

    def sigmoid(self):
        sig = 1 / (1 + np.exp(-self.data))
        out = Tensor(sig, requires_grad=self.requires_grad)
        out._prev = [self]
        out._op = 'sigmoid'
        return out

    def tanh(self):
        out = Tensor(np.tanh(self.data), requires_grad=self.requires_grad)
        out._prev = [self]
        out._op = 'tanh'
        return out

    def softmax(self):
        exps = np.exp(self.data - np.max(self.data, axis=-1, keepdims=True))
        out = Tensor(exps / np.sum(exps, axis=-1, keepdims=True), requires_grad=self.requires_grad)
        out._prev = [self]
        out._op = 'softmax'
        return out

    # 链式反向传播
    def backward(self, grad=None):
        if not self.requires_grad:
            return
        if grad is None:
            # print("grad is None")
            grad = np.zeros_like(self.data)
        # Check if grad contains NaN or Inf
        if np.isnan(grad).any() or np.isinf(grad).any():
            print("Warning: NaN or Inf in grad")
        # print(self.grad.shape)
        # print(grad.shape)
        if (self.grad.shape != grad.shape):
            if grad.size == 0:
                grad = np.zeros_like(self.grad)
            elif grad.ndim == 0:  # 如果是标量，就保持原样
                grad = grad
            elif self.grad.ndim == 1:
                # print(grad.shape)
                grad = grad.mean(axis=0)
        
        # print(grad)
        self.grad += grad
        # 打印每层的梯度
        # print(f"Gradient of {self._op}: {self.grad}")

        # 链式规则
        if self._op == 'add':
            # print("add")
            # print(self._prev)
            self._prev[0].backward(self.grad)
            if len(self._prev) > 1:
                self._prev[1].backward(self.grad)

        elif self._op == 'sub':
            # print("sub")
            self._prev[0].backward(self.grad)
            if len(self._prev) > 1:
                self._prev[1].backward(-self.grad)

        elif self._op == 'mul':
            # print("mul")
            self._prev[0].backward(self.grad * self._prev[1].data)
            if len(self._prev) > 1:
                self._prev[1].backward(self.grad * self._prev[0].data)

        elif self._op == 'div':
            # print("div")
            self._prev[0].backward(self.grad / self._prev[1].data)
            if len(self._prev) > 1:
                self._prev[1].backward(-self.grad * self._prev[0].data / (self._prev[1].data ** 2))

        elif self._op == 'pow':
            # print("pow")
            self._prev[0].backward(self.grad * self._power * self._prev[0].data ** (self._power - 1))

        elif self._op == 'neg':
            # print("neg")
            self._prev[0].backward(-self.grad)

        elif self._op == 'matmul':
            # print("matmul")
            # print("0")
            self._prev[0].backward(np.dot(self.grad, self._prev[1].data.T))
            # print("1")
            self._prev[1].backward(np.dot(self._prev[0].data.T, self.grad))
            
        elif self._op == 'relu':
            # print("relu")
            relu_grad = self.grad * (self.data > 0)
            self._prev[0].backward(relu_grad)

        elif self._op == 'sigmoid':
            # print("sigmoid")
            sigmoid_grad = self.grad * (self.data * (1 - self.data))
            self._prev[0].backward(sigmoid_grad)

        elif self._op == 'tanh':
            tanh_grad = self.grad * (1 - self.data ** 2)
            self._prev[0].backward(tanh_grad)

        elif self._op == 'softmax':
            # Softmax backward pass is more complex and typically handled with cross-entropy loss
            self._prev[0].backward(self.grad)
            
        elif self._op == 'mse':
            # print(self.grad)
            self._prev[0].backward(self.grad)
            
        elif self._op == 'cross_entropy':
            self._prev[0].backward(self.grad)

    # 清零梯度
    def zero_grad(self):
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)
            
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def T(self):
        return Tensor(self.data.T, requires_grad=self.requires_grad)

    def mean(self):
        return Tensor(np.mean(self.data), requires_grad=self.requires_grad)