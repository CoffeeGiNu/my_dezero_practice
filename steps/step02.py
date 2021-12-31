class Variable:
    def __init__(self, data):
        self.data = data
        
class Function:
    def __call__(self, input):
        x = self.data
        y = self.forward(x) # 具体的な計算はforwardメソッドで行う
        output = Variable(y)
        return output
    
    def forward(self, x):
        raise NotImplementedError()

def Square(Function):
    def forward(self, x):
        return x ** 2