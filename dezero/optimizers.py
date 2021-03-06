import numpy as np
from dezero import cuda
from typing import NoReturn

class Optimizer:
    def __init__(self):
        self.target = None
        self.hooks = []
    
    def setup(self, target):
        self.target = target
        return self
    
    def update(self):
        # None以外のパラメータをリストにまとめる
        params = [p for p in self.target.params() if p.grad is not None]

        # 前処理(オプション)
        for f in self.hooks:
            f(params)
        
        # パラメータの更新
        for param in params:
            self.update_one(param)
    
    def update_one(self, f):
        raise NotImplementedError()

    def add_hook(self, f):
        self.hooks.append(f)

class SGD(Optimizer):
    def __init__(self, lr=0.01):
        super().__init__()
        self.lr = lr
    
    def update_one(self, param):
        param.data -= self.lr * param.grad.data

class MomentumSGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.vs = {}

    def update_one(self, param):
        xp = cuda.get_array_module(param)
        v_key = id(param)
        if v_key not in self.vs:
            self.vs[v_key] = xp.zeros_like(param.data)
        
        v = self.vs[v_key]
        v = v * self.momentum
        v = v - self.lr * param.grad.data
        param.data = param.data + v