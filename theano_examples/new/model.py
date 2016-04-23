import numpy
import theano
import theano.tensor as T


class SoftMax:
    def __init__(self, n_input, n_output):
        # initialize with 0 the weights as a matrix of shape (n_in, n_out)
        self.w = theano.shared(
            value=numpy.zeros((n_input, n_output), dtype=theano.config.floatX),
            name='w',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros((n_output,), dtype=theano.config.floatX),
            name='b',
            borrow=True
        )
        # input and output
        self.x = T.matrix('x')
        self.y = T.ivector('y')
        # hyperplane
        p_y_given_x = T.nnet.softmax(T.dot(self.x, self.w) + self.b)
        # probability is maximal
        y_pred = T.argmax(p_y_given_x, axis=1)
        # error
        self.error = T.mean(T.neq(y_pred, self.y))
        # cost
        self.cost = -T.mean(T.log(p_y_given_x)[T.arange(self.y.shape[0]), self.y])


class HiddenLayer:
    def __init__(self, n_input, n_output, activation=T.tanh):
        weight_max = numpy.sqrt(6. / (n_input + n_output))
        if activation == theano.tensor.nnet.sigmoid:
            weight_max *= 4
        rng = numpy.random.RandomState(1234)
        self.w = theano.shared(
            value=rng.uniform(low=-weight_max, high=weight_max, size=(n_input, n_output)),
            name='w',
            borrow=True
        )
        self.b = theano.shared(
            value= numpy.zeros((n_output,)),
            name='b',
            borrow=True
        )
        self.p_y_given_x = T.dot(input, self.w) + self.b
        if activation is not None:
            self.p_y_given_x = activation(self.p_y_given_x)


