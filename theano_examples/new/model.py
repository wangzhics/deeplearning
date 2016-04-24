import numpy
import theano
import theano.tensor as T


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
    def __init__(self, x, y, n_x, n_y, activation=T.tanh):
        weight_max = numpy.sqrt(6. / (n_x + n_y))
        if activation == theano.tensor.nnet.sigmoid:
            weight_max *= 4
        rng = numpy.random.RandomState(1234)
        self.w = theano.shared(
            value=rng.uniform(low=-weight_max, high=weight_max, size=(n_input, n_output)),
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
        # calculate
        self.y_given_x = T.dot(self.x, self.w) + self.b
        if activation is not None:
            self.y_given_x = activation(self.y_given_x)


