import numpy
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

# the parameters
x = T.matrix("x")
y = T.vector("y")
w = theano.shared(numpy.zeros(2), name="w")  # 特征值的权值
b = theano.shared(0., name="b")  # 常数
c = theano.shared(0., name="c")  # 损失函数


def inspect_inputs(i, node, fn):
    pass


def inspect_outputs(i, node, fn):
    if i == 0:
        print("Cost is ", c.get_value(), " Weight are ", w.get_value(), b.get_value())

# build model
weight_x = T.dot(x, w) + b  # 特征矩阵乘以权值
sigmoid_y = 1 / (1 + T.exp(-weight_x))  # 转为Sigmoid函数
prediction = sigmoid_y > 0.5  # 实际预测值
cost = -y * T.log(sigmoid_y) - (1-y) * T.log(1-sigmoid_y)  # 损失函数
total_cost = cost.mean() + 0.01 * (w ** 2).sum()  # 加上L2正则化，使total_cost最小
cost_print = theano.printing.Print("cost is")(total_cost)
gw, gb = T.grad(total_cost, [w, b])
train = theano.function(inputs=[x, y],
                        outputs=[prediction, total_cost],
                        updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb), (c, total_cost)),
                        mode=theano.compile.MonitorMode(pre_func=inspect_inputs, post_func=inspect_outputs))
predict = theano.function(inputs=[x], outputs=prediction)


# load data
xs = []
ys = []
with open('testSet.txt') as f:
    for line in f.readlines():
        line_array = line.strip().split('\t')
        xs.append([float(line_array[0]), float(line_array[1])])
        ys.append(float(line_array[2]))

# begin train
print("Before Training Weight are: ", w.get_value(), b.get_value())
training_steps = 1000
for i in range(training_steps):
    predict, error = train(xs, ys)
print("After Training Weight are: ", w.get_value(), b.get_value())

print(xs)
print(ys)
print(predict)


# paint all spot
plt_0_x_list = []
plt_0_y_list = []
plt_1_x_list = []
plt_1_y_list = []
for i in range(len(xs)):
    if ys[i] == 0:
        plt_0_x_list.append(xs[i][0])
        plt_0_y_list.append(xs[i][1])
    else:
        plt_1_x_list.append(xs[i][0])
        plt_1_y_list.append(xs[i][1])
plt.scatter(plt_0_x_list, plt_0_y_list, c="red", s=20)
plt.scatter(plt_1_x_list, plt_1_y_list, c="yellow", marker="s", s=20)
# paint boundary
weights = [w.get_value()[0], w.get_value()[1], b.get_value()]
x_list = numpy.arange(-3.5, 3.5, 0.1)
y_list = []
for x in x_list:
    y = (0 - weights[2] - weights[0] * x) / weights[1]
    y_list.append(y)
plt.plot(x_list, y_list, color="green", linestyle="solid")

plt.show()

