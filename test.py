# import numpy as np
# import dezero

# x = Variable(np.array(1.0))
# print(x)

# def sphere(x, y):
#     z = x ** 2 + y ** 2
#     return z


# def matyas(x, y):
#     z = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
#     return z


# def goldstein(x, y):
#     z = (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) * \
#         (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
#     return z


# x = Variable(np.array(1.0))
# y = Variable(np.array(1.0))
# z = goldstein(x, y)  # sphere(x, y) / matyas(x, y)
# z.backward()
# print(x.grad, y.grad)

# v = 'test'
# v_name = 'test_name'
# v_data = '100'

# x = 'test' if v is None else v
# print(x)

# verbose = False
# if v_name is None:
#     name = ''
# else:
#     # if verbose and v_data is not None:
#     #     if v_name is not None:
#     #         name += ': '
#     #     name += str(v.shape) + ' ' + str(v.dtype)
#     print(v)
#     print(name)

# if '__file__' in globals():
#     import os, sys
#     sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# import numpy as np
# import matplotlib.pyplot as plt
# from dezero import Variable
# import dezero.functions as F

# x = Variable(np.linspace(-7, 7, 200))
# y = F.sin(x)
# y.backward(create_graph=True)

# logs = [y.data]

# for i in range(3):
#     logs.append(x.grad.data)
#     gx = x.grad
#     x.cleargrad()
#     gx.backward(create_graph=True)

# labels = ["y=sin(x)", "y'", "y''", "y'''"]
# for i, v in enumerate(logs):
#     plt.plot(x.data, logs[i], label=labels[i])
# plt.legend(loc='lower right')
# plt.show()

# x = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
# print(x)
# x = dezero.as_variable(x)
# print(x)
# x = x.reshape(2)
# print(x)

# if '__file__' in globals():
#     import os, sys
#     sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# import numpy as np
# import matplotlib.pyplot as plt
# from dezero import Variable
# import dezero.functions as F

# # Generate toy dataset
# np.random.seed(0)
# x = np.random.rand(100, 1)
# y = 5 + 2 * x + np.random.rand(100, 1)
# x, y = Variable(x), Variable(y)

# W = Variable(np.zeros((1, 1)))
# b = Variable(np.zeros(1))


# def predict(x):
#     y = F.matmul(x, W) + b
#     return y


# def mean_squared_error(x0, x1):
#     diff = x0 - x1
#     return F.sum(diff ** 2) / len(diff)


# lr = 0.1
# iters = 100

# for i in range(iters):
#     y_pred = predict(x)
#     loss = mean_squared_error(y, y_pred)

#     W.cleargrad()
#     b.cleargrad()
#     loss.backward()

#     # Update .data attribute (No need grads when updating params)
#     W.data -= lr * W.grad.data
#     b.data -= lr * b.grad.data
#     print(W, b, loss)


# # Plot
# plt.scatter(x.data, y.data, s=10)
# plt.xlabel('x')
# plt.ylabel('y')
# y_pred = predict(x)
# plt.plot(x.data, y_pred.data, color='r')
# plt.show()

# import numpy as np
# from dezero import Variable
# import dezero.functions as F

# np.random.seed(0)
# x = np.random.rand(100, 1)
# y = 5 + 2 * x + np.random.rand(100, 1)
# x, y = Variable(x), Variable(y)

# W = Variable(np.zeros((1, 1)))
# b = Variable(np.zeros(1))

# def predict(x):
#     y = F.matmul(x, W) + b
#     return y

# def mean_squared_error(x0, x1):
#     diff = x0 - x1
#     return F.sum(diff ** 2) / len(diff)

# lr = 0.1
# iters = 100

# for i in range(iters):
#     y_pred = predict(x)
#     loss = mean_squared_error(y, y_pred)

#     W.cleargrad()
#     b.cleargrad()
#     loss.backward()

#     # Update .data attribute (No need grads when updating params)
#     W.data -= lr * W.grad.data
#     b.data -= lr * b.grad.data
#     print(W, b, loss)
# if '__file__' in globals():
#     import os, sys
#     sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# import numpy as np
# from dezero import Variable
# import dezero.functions as F

# x = Variable(np.array([1, 2, 3, 4, 5, 6]))
# y = F.sum(x)
# y.backward()
# print(y)
# print(x.grad)

# x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
# y = F.sum(x)
# y.backward()
# print(y)
# print(x.grad)

# x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
# y = F.sum(x, axis=0)
# y.backward()
# print(y)
# print(x.grad)

# x = Variable(np.random.randn(2, 3, 4, 5))
# y = x.sum(keepdims=True)
# print(y.shape)

# if '__file__' in globals():
#     import os, sys
#     sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# import numpy as np
# import matplotlib.pyplot as plt
# from dezero import Variable
# import dezero.functions as F


# np.random.seed(0)
# x = np.random.rand(100, 1)
# y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

# I, H, O = 1, 10, 1
# W1 = Variable(0.01 * np.random.randn(I, H))
# b1 = Variable(np.zeros(H))
# W2 = Variable(0.01 * np.random.randn(H, O))
# b2 = Variable(np.zeros(O))


# def predict(x):
#     y = F.linear_simple(x, W1, b1)
#     y = F.sigmoid_simple(y)
#     y = F.linear_simple(y, W2, b2)
#     return y


# lr = 0.2
# iters = 10000

# for i in range(iters):
#     y_pred = predict(x)
#     loss = F.mean_squared_error(y, y_pred)

#     W1.cleargrad()
#     b1.cleargrad()
#     W2.cleargrad()
#     b2.cleargrad()
#     loss.backward()

#     W1.data -= lr * W1.grad.data
#     b1.data -= lr * b1.grad.data
#     W2.data -= lr * W2.grad.data
#     b2.data -= lr * b2.grad.data
#     if i % 1000 == 0:
#         print(loss)


# # Plot
# plt.scatter(x, y, s=10)
# plt.xlabel('x')
# plt.ylabel('y')
# t = np.arange(0, 1, .01)[:, np.newaxis]
# y_pred = predict(t)
# plt.plot(t, y_pred.data, color='r')
# plt.show()

# if '__file__' in globals():
#     import os, sys
#     sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# import numpy as np
# import dezero.functions as F
# import dezero.layers as L


# np.random.seed(0)
# x = np.random.rand(100, 1)
# y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

# l1 = L.Linear(10)
# l2 = L.Linear(1)


# def predict(x):
#     y = l1(x)
#     y = F.sigmoid(y)
#     y = l2(y)
#     return y


# lr = 0.2
# iters = 10000

# for i in range(iters):
#     y_pred = predict(x)
#     loss = F.mean_squared_error(y, y_pred)

#     l1.cleargrads()
#     l2.cleargrads()
#     loss.backward()

#     for l in [l1, l2]:
#         for p in l.params():
#             p.data -= lr * p.grad.data
#     if i % 1000 == 0:
#         print(l1.__dict__['W'])
#         print(loss)
# if '__file__' in globals():
#     import os, sys
#     sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# import numpy as np
# from dezero import Model
# import dezero.layers as L
# import dezero.functions as F


# np.random.seed(0)
# x = np.random.rand(100, 1)
# y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

# # Hyperparameters
# lr = 0.2
# max_iter = 10000
# hidden_size = 10

# # Model definition
# class TwoLayerNet(Model):
#     def __init__(self, hidden_size, out_size):
#         super().__init__()
#         self.l1 = L.Linear(hidden_size)
#         self.l2 = L.Linear(out_size)

#     def forward(self, x):
#         y = F.sigmoid(self.l1(x))
#         y = self.l2(y)
#         return y


# model = TwoLayerNet(hidden_size, 1)

# for i in range(max_iter):
#     y_pred = model(x)
#     loss = F.mean_squared_error(y, y_pred)

#     model.cleargrads()
#     loss.backward()

#     for p in model.params():
#         p.data -= lr * p.grad.data
#     if i % 1000 == 0:
#         print(loss)

# if '__file__' in globals():
#     import os, sys
#     sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# import numpy as np
# from dezero import optimizers
# import dezero.functions as F
# from dezero.models import MLP


# np.random.seed(0)
# x = np.random.rand(100, 1)
# y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

# lr = 0.2
# max_iter = 10000
# hidden_size = 10

# model = MLP((hidden_size, 1))
# optimizer = optimizers.SGD(lr).setup(model)

# for i in range(max_iter):
#     y_pred = model(x)
#     loss = F.mean_squared_error(y, y_pred)

#     model.cleargrads()
#     loss.backward()

#     optimizer.update()
#     if i % 1000 == 0:
#         print(loss)

# if '__file__' in globals():
#     import os, sys
#     sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# import numpy as np
# np.random.seed(0)
# from dezero import Variable, as_variable
# import dezero.functions as F
# from dezero.models import MLP


# def softmax1d(x):
#     x = as_variable(x)
#     y = F.exp(x)
#     sum_y = F.sum(y)
#     return y / sum_y


# model = MLP((10, 3))

# # x = Variable(np.array([[0.2, -0.4]]))
# # y = model(x)
# # p = softmax1d(y)
# # print(y)
# # print(p)

# x = np.array([[0.2, -0.4], [0.3, 0.5], [1.3, -3.2], [2.1, 0.3]])
# t = np.array([2, 0, 1, 0])

# y = model(x)
# # print(y)
# p = F.softmax_simple(y)
# print(p)

# loss = F.softmax_cross_entropy_simple(y, t)
# loss.backward()
# print(loss)


# t = [1, 2, 3]
# x = iter(t)
# print(next(x))
# print(next(x))
# print(next(x))
# print(next(x))

# import numpy as np

# batch_size = 500
# i = 0
# index = np.random.permutation(2000)

# batch_index = index[i * batch_size:(i + 1) * batch_size]
# print(batch_index)
# import numpy as np
# pred = np.array([[0], [1], [0], [1], [0], [1]])
# teach = np.array([[1], [1], [1], [1], [1], [1]])
# result = (pred == teach)
# print(result)

# import cupy

# import numpy
# a = numpy.array([[1,2],[3,4],[5,6]])
# print (a[:, 0])  # array([1, 3, 5])
# import numpy as np
# from dezero.models import VGG16
# model = VGG16(pretrained=True)
# x = np.random.randn(1, 3, 224, 224).astype(np.float32)
# model.plot(x)

# if '__file__' in globals():
#     import os, sys
#     sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# import numpy as np
# import matplotlib.pyplot as plt
# import dezero
# from dezero import Model
# import dezero.functions as F
# import dezero.layers as L

# # Hyperparameters
# max_epoch = 100
# hidden_size = 100
# bptt_length = 30

# train_set = dezero.datasets.SinCurve(train=True)
# seqlen = len(train_set)


# class SimpleRNN(Model):
#     def __init__(self, hidden_size, out_size):
#         super().__init__()
#         self.rnn = L.RNN(hidden_size)
#         self.fc = L.Linear(out_size)

#     def reset_state(self):
#         self.rnn.reset_state()

#     def __call__(self, x):
#         h = self.rnn(x)
#         y = self.fc(h)
#         return y

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib.pyplot as plt
import dezero
from dezero import Model
from dezero import SeqDataLoader
import dezero.functions as F
import dezero.layers as L


max_epoch = 100
batch_size = 30
hidden_size = 100
bptt_length = 30

train_set = dezero.datasets.SinCurve(train=True)
dataloader = SeqDataLoader(train_set, batch_size=batch_size)
seqlen = len(train_set)


class BetterRNN(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.rnn = L.LSTM(hidden_size)
        self.fc = L.Linear(out_size)

    def reset_state(self):
        self.rnn.reset_state()

    def __call__(self, x):
        y = self.rnn(x)
        y = self.fc(y)
        return y

model = BetterRNN(hidden_size, 1)
optimizer = dezero.optimizers.Adam().setup(model)

for epoch in range(max_epoch):
    model.reset_state()
    loss, count = 0, 0

    for x, t in dataloader:
        y = model(x)
        loss += F.mean_squared_error(y, t)
        count += 1

        if count % bptt_length == 0 or count == seqlen:
            model.cleargrads()
            loss.backward()
            loss.unchain_backward()
            optimizer.update()
    avg_loss = float(loss.data) / count
    print('| epoch %d | loss %f' % (epoch + 1, avg_loss))

# Plot
xs = np.cos(np.linspace(0, 4 * np.pi, 1000))
model.reset_state()
pred_list = []

with dezero.no_grad():
    for x in xs:
        x = np.array(x).reshape(1, 1)
        y = model(x)
        pred_list.append(float(y.data))

plt.plot(np.arange(len(xs)), xs, label='y=cos(x)')
plt.plot(np.arange(len(xs)), pred_list, label='predict')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()