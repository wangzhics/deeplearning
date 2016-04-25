import numpy
import theano
import theano.tensor as T
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.pool import pool_2d


class SoftMax:
    def __init__(self, x, y, n_x, n_y):
        # initialize with 0 the weights as a matrix of shape (n_in, n_out)
        self.w = theano.shared(
            value=numpy.zeros((n_x, n_y), dtype=theano.config.floatX),
            name='w',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros((n_y,), dtype=theano.config.floatX),
            name='b',
            borrow=True
        )
        self.params = [self.w, self.b]
        # save x, y
        self.x = x
        self.y = y
        # calculate
        p_y_given_x = T.nnet.softmax(T.dot(self.x, self.w) + self.b)
        # probability is maximal
        y_pred = T.argmax(p_y_given_x, axis=1)
        # error
        self.error = T.mean(T.neq(y_pred, self.y))
        # cost
        self.cost = -T.mean(T.log(p_y_given_x)[T.arange(self.y.shape[0]), self.y])


class HiddenLayer:
    def __init__(self, rng, x, y, n_x, n_y, activation=T.tanh):
        weight_max = numpy.sqrt(6. / (n_x + n_y))
        if activation == theano.tensor.nnet.sigmoid:
            weight_max *= 4
        self.w = theano.shared(
            value=rng.uniform(low=-weight_max, high=weight_max, size=(n_x, n_y)),
            name='w',
            borrow=True
        )
        self.b = theano.shared(
            value= numpy.zeros((n_y,), dtype=theano.config.floatX),
            name='b',
            borrow=True
        )
        self.params = [self.w, self.b]
        # save x, y
        self.x = x
        self.y = y
        # calculate the output
        self.y_given_x = T.dot(self.x, self.w) + self.b
        if activation is not None:
            self.y_given_x = activation(self.y_given_x)


class LeNetConvPoolLayer:
    def __init__(self, rng, input, input_shape, filter_shape, pool_shape=(2, 2)):
        """
        构造一个卷积池化层，包含一个卷积层和一个池化层
        :param input: 输入的图像
        :param input_shape: 输入的图像形状：(batch_size, image_channel, image_weight, image_height)
        :param filter_shape: 过滤器的形状：(filter_count, image_channel, filter_weight, filter_height)
        :param pool_shape: 池化器的形状
        :return:
        """
        #
        assert input_shape[1] == filter_shape[1]
        self.input = input
        self.input_shape = input_shape
        self.filter_shape = filter_shape
        self.pool_shape = pool_shape
        # 每张图像的输入输出
        n_in = numpy.prod(input_shape[1:])
        n_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) // numpy.prod(pool_shape))
        weight_max = numpy.sqrt(6. / (n_in + n_out))
        self.w = theano.shared(
            numpy.asarray(
                rng.uniform(low=-weight_max, high=weight_max, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )
        self.b = theano.shared(numpy.zeros((filter_shape[0],), dtype=theano.config.floatX), borrow=True)
        self.conv_out = conv2d(
            input=self.input,
            filters=self.w,
            filter_shape=self.filter_shape,
            image_shape=self.input_shape
        )
        self.pool_out = pool_2d(
            input=self.conv_out,
            ds=pool_shape,
            ignore_border=True
        )
        self.output = T.tanh(self.pool_out + self.b.dimshuffle('x', 0, 'x', 'x'))