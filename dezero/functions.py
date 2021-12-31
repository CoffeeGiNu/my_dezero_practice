import numpy as np
import dezero
from dezero import cuda, utils
from dezero.core import Function, Variable, as_variable, as_array

class Sin(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        return xp.sin(x)

    def backward(self, gy):
        x, = self.inputs
        return gy * cos(x)

def sin(x):
    return Sin()(x)

class Cos(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        return xp.cos(x)
    
    def backward(self, gy):
        x, = self.inputs
        return gy * -sin(x)

def cos(x):
    return Cos()(x)

class Tanh(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        return xp.tanh(x)
    
    def backward(self, gy):
        y = self.outputs[0]()
        return gy * (1 - y ** 2)

def tanh(x):
    return Tanh()(x)

class Exp(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        return xp.exp(x)

    def backward(self, gy):
        y = self.outputs[0]()  # weakref
        gx = gy * y
        return gx

def exp(x):
    return Exp()(x)

class Log(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.log(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy / x
        return gx

def log(x):
    return Log()(x)

class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape
    
    def forward(self, x):
        self.x_shape = x.shape
        return x.reshape(self.shape)

    def backward(self, gy):
        return reshape(gy, self.x_shape)

def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)

class Transpose(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        return xp.transpose(x)
    
    def backward(self, gy):
        return transpose(gy)

def transpose(x):
    return Transpose()(x)

class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y
    
    def backward(self, gy):
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis,
                                        self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
        return gx

def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)

class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape
    
    def forward(self, x):
        xp = cuda.get_array_module(x)
        self.x_shape = x.shape
        return xp.broadcast_to(x, self.shape)

    def backward(self, gy):
        return sum_to(gy, self.x_shape)

def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)

class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape
    
    def forward(self, x):
        self.x_shape = x.shape
        return utils.sum_to(x, self.shape)
    
    def backward(self, gy):
        return broadcast_to(gy, self.x_shape)

def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)

class MatMul(Function):
    def forward(self, x, W):
        return x.dot(W)
    
    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW

def matmul(x, W):
    return MatMul()(x, W)

def linear_simple(x, W, b=None):
    # x, W = as_variable(x), as_variable(W)
    t = matmul(x, W)
    if b is None:
        return t
    y = t + b
    t.data = None
    return y

class Linear(Function):
    def forward(self, x, W, b):
        y = x.dot(W)
        if b is not None:
            y += b
        return y

    def backward(self, gy):
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW, gb

def linear(x, W, b=None):
    return Linear()(x, W, b)

def sigmoid_simple(x):
    x = as_variable(x)
    y = 1 / (1 + exp(-x))
    return y

class Sigmoid(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        # y = 1 / (1 + xp.exp(-x)) # 元々の実装方法
        return xp.tanh(x * 0.5) * 0.5 + 0.5  # より良い実装方法

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y * (1 - y)
        return gx

def sigmoid(x):
    return Sigmoid()(x)

class GetItem(Function):
    def __init__(self, slices):
        self.slices = slices
    
    def forward(self, x):
        y = x[self.slices]
        return y
    
    def backward(self, gy):
        x, = self.inputs
        f = GetItemGrad(self.slices, x.shape)
        return f(gy)

class GetItemGrad(Function):
    def __init__(self, slices, in_shape):
        self.slices = slices
        self.in_shape = in_shape
    
    def forward(self, gy):
        xp = cuda.get_array_module(gy)
        gx = xp.zeros(self.in_shape, dtype=gy.dtype)
        xp.add.at(gx, self.slices, gy)
        return gx
    
    def backward(self, ggx):
        return get_item(ggx, self.slices)

def get_item(x, slices):
    f = GetItem(slices)
    return f(x)

def softmax_simple(x, axis=1):
    x = as_variable(x)
    x = exp(x)
    sum_x = sum(x, axis=axis, keepdims=True)
    y = x / sum_x
    return y

class Softmax(Function):
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = x - x.max(axis=self.axis, keepdims=True)
        y = xp.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = y * gy
        sumdx = gx.sum(axis=self.axis, keepdims=True)
        gx -= y * sumdx
        return gx

def softmax(x, axis=1):
    return Softmax(axis)(x)

def softmax_cross_entropy_simple(x, t):
    xp = cuda.get_array_module(x)
    x, t = as_variable(x), as_variable(t)
    N = x.shape[0] # いくつのバッチデータが入力されたか
    p = softmax(x) # ここでソフトマックス関数を適用
    p = clip(p, 1e-15, 1.0)  # log(0)を防ぐためにle-15以上にする
    log_p = log(p) # log(p)を計算
    log_pt = log_p[xp.arange(N), t.data] # バッチごとに正解と出力結果との誤差を計算
    y = -1 * sum(log_pt) / N # 全体の誤差の平均を計算
    return y

class ReLU(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.maximum(x, 0.0) # 0以上ならばそのまま出力/0以下ならば0を出力
        return y

    def backward(self, gy):
        x, = self.inputs
        mask = x.data > 0
        gx = gy * mask
        return gx

def relu(x):
    return ReLU()(x)

class MeanSquaredError(Function):
    def forward(self, x0, x1):
        diff = x0 - x1
        return (diff ** 2).sum() / len(diff)
    
    def backward(self, gy):
        x0, x1 = self.inputs
        diff = x0 - x1
        # gy = broadcast_to(gy, diff.shape)
        gx0 = gy * diff * (2.0 / len(diff))
        gx1 = -gx0
        return gx0, gx1

def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)

def accuracy(y, t):
    y, t = as_variable(y), as_variable(t)

    # 予測結果の最大値のインデックスを求める
    pred = y.data.argmax(axis=1).reshape(t.shape)
    # 予測結果と正解を比較
    result = (pred == t.data)
    # 正解率を算出
    acc = result.mean()
    # ndarrayインスタンスに変換
    return Variable(as_array(acc))

def dropout(x, dropout_ratio=0.5):
    x = as_variable(x)

    if dezero.Config.train:
        xp = cuda.get_array_module(x)
        mask = xp.random.randn(*x.shape) > dropout_ratio
        scale = xp.array(1.0 - dropout_ratio).astype(x.dtype)
        y = x * mask / scale
        return y
    else:
        return x

class Clip(Function):
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.clip(x, self.x_min, self.x_max)
        return y

    def backward(self, gy):
        x, = self.inputs
        mask = (x.data >= self.x_min) * (x.data <= self.x_max)
        gx = gy * mask
        return gx

def clip(x, x_min, x_max):
    return Clip(x_min, x_max)(x)

class Max(Function):
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        y = x.max(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        x = self.inputs[0]
        y = self.outputs[0]() # weakref

        shape = utils.max_backward_shape(x, self.axis)
        gy = reshape(gy, shape)
        y = reshape(y, shape)
        cond = (x.data == y.data) # 取り出された部分を確認
        gy = broadcast_to(gy, cond.shape) # condをgyの形に合わせる
        
        return gy * cond

class Min(Max):
    def forward(self, x):
        y = x.min(axis=self.axis, keepdims=self.keepdims)
        return y

def max(x, axis=None, keepdims=False):
    return Max(axis, keepdims)(x)

def min(x, axis=None, keepdims=False):
    return Min(axis, keepdims)(x)


from dezero.functions_conv import conv2d
from dezero.functions_conv import deconv2d
from dezero.functions_conv import conv2d_simple
from dezero.functions_conv import im2col
from dezero.functions_conv import col2im
from dezero.functions_conv import pooling_simple
from dezero.functions_conv import pooling
from dezero.functions_conv import average_pooling
from dezero.core import add
from dezero.core import sub
from dezero.core import rsub
from dezero.core import mul
from dezero.core import div
from dezero.core import neg
from dezero.core import pow