import numpy
import theano
import theano.tensor as T
from theano_examples.new.model import SoftMax
from theano_examples.new.train import load_data, sgd_train


if __name__ == '__main__':
    # load data
    train_set, valid_set, test_set = load_data()
    train_set_x, train_set_y = train_set
    valid_set_x, valid_set_y = valid_set
    # build model
    index = T.lscalar()
    batch_size = 600
    x = T.matrix('x')
    y = T.ivector('y')
    soft_max = SoftMax(x, y, 28*28, 10)
    g_w = T.grad(cost=soft_max.cost, wrt=soft_max.w)
    g_b = T.grad(cost=soft_max.cost, wrt=soft_max.b)
    learning_rate = 0.15
    updates = [(soft_max.w, soft_max.w - learning_rate * g_w),
               (soft_max.b, soft_max.b - learning_rate * g_b)]
    index = T.lscalar()
    batch_size = 600
    # train model
    train_model = theano.function(
        inputs=[index],
        outputs=soft_max.cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # validate model
    validate_model = theano.function(
        inputs=[index],
        outputs=soft_max.error,
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # remember the best
    best_w = soft_max.w.get_value(borrow=False)
    best_b = soft_max.b.get_value(borrow=False)

    def finish_once(improve):
        if improve:
            best_w = soft_max.w.get_value(borrow=False)
            best_b = soft_max.b.get_value(borrow=False)
    sgd_train(train_set, valid_set, train_model, validate_model, finish_once, default_batch_size=batch_size)
    soft_max.w.set_value(best_w)
    soft_max.b.set_value(best_b)
