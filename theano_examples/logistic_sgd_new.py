import numpy
import theano
import theano.tensor as T
from theano_examples.data import load_data


def shared_dataset(data_xy, borrow=True):
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
    return shared_x, shared_y


class SoftMax:
    def __init__(self, n_in, n_out):
        # initialize with 0 the weights as a matrix of shape (n_in, n_out)
        self.w = theano.shared(
            value=numpy.zeros((n_in, n_out), dtype=theano.config.floatX),
            name='w',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros((n_out,), dtype=theano.config.floatX),
            name='b',
            borrow=True
        )
        # input and output
        x = T.matrix('x')
        y = T.ivector('y')
        # hyperplane
        p_y_given_x = T.nnet.softmax(T.dot(x, self.w) + self.b)
        # probability is maximal
        y_pred = T.argmax(p_y_given_x, axis=1)
        # cost
        cost = -T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]), y])
        # error
        error = T.mean(T.neq(y_pred, y))
        # constant value
        self.learning_rate = 0.15
        self.batch_size = 600
        # grad
        g_w = T.grad(cost=cost, wrt=self.w)
        g_b = T.grad(cost=cost, wrt=self.b)
        updates = [(self.w, self.w - self.learning_rate * g_w),
                   (self.b, self.b - self.learning_rate * g_b)]

        index = T.lscalar()
        # train model
        self.train_model = theano.function(inputs=[x, y], outputs=cost, updates=updates, allow_input_downcast=True)
        # validate model
        self.validate_model = theano.function(inputs=[x, y], outputs=error, allow_input_downcast=True)
        # test model
        self.test_model = theano.function(inputs=[x, y], outputs=error, allow_input_downcast=True)
        # get y
        index = T.lscalar()
        print_y = theano.printing.Print('train y ')(y[index * self.batch_size: (index + 1) * self.batch_size])
        self.get_y = theano.function(inputs=[y, index],
                                     outputs=print_y)

    def _train_sgd(self, train_set_x, train_set_y):
        x = train_set_x.get_value(borrow=True)
        y = train_set_y.get_value(borrow=True)
        batch_count = x.shape[0] // self.batch_size
        # train by batch
        for i in range(batch_count):
            x_batch = x[i * self.batch_size: (i + 1) * self.batch_size]
            y_batch = y[i * self.batch_size: (i + 1) * self.batch_size]
            self.train_model(x_batch, y_batch)

    def _validate(self, valid_set_x, valid_set_y):
        x = valid_set_x.get_value(borrow=True)
        y = valid_set_y.get_value(borrow=True)
        batch_count = x.shape[0] // self.batch_size
        # validate by batch
        validation_losses = []
        for i in range(batch_count):
            x_batch = x[i * self.batch_size: (i + 1) * self.batch_size]
            y_batch = y[i * self.batch_size: (i + 1) * self.batch_size]
            validation_losses.append(self.validate_model(x_batch, y_batch))
        return numpy.mean(validation_losses)


    def test(self, test_set):
        test_set_x, test_set_y = shared_dataset(test_set)
        return self._validate(test_set_x, test_set_y)

    def train(self, train_set, valid_set):
        train_set_x, train_set_y = shared_dataset(train_set)
        valid_set_x, valid_set_y = shared_dataset(valid_set)
        # stop when
        best_valid_error_rate = 1
        best_w = self.w.get_value(borrow=False)
        best_b = self.b.get_value(borrow=False)
        running = True
        # early stop
        not_improve = 0
        i = 0
        while running:
            self._train_sgd(train_set_x, train_set_y)
            valid_error_rate = self._validate(valid_set_x, valid_set_y)
            if valid_error_rate < best_valid_error_rate:
                best_valid_error_rate = valid_error_rate
                best_w = self.w.get_value(borrow=False)
                best_b = self.b.get_value(borrow=False)
                not_improve = 0
            else:
                not_improve += 1
            if not_improve > 5:
                running = False
                self.w.set_value(best_w, borrow=True)
                self.b.set_value(best_b, borrow=True)
            # debug info
            # print("train step %d , valid_error_rate %f%%" %(i, valid_error_rate * 100))
            # i += 1


if __name__ == '__main__':
    soft_max = SoftMax(28*28, 10)
    # Load the dataset
    train_set, valid_set, test_set = load_data()
    # train
    soft_max.train(train_set, valid_set)
    # test
    test_error_rate = soft_max.test(test_set)
    print("final test error rate %f" % (test_error_rate * 100))

