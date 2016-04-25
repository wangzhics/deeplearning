import numpy
import theano
import theano.tensor as T
from theano_examples.new.model import SoftMax, HiddenLayer
from theano_examples.new.train import load_data, sgd_train


if __name__ == '__main__':
    # load data
    train_set, valid_set, test_set = load_data()
    train_set_x, train_set_y = train_set
    valid_set_x, valid_set_y = valid_set
    # build model
    index = T.lscalar()
    batch_size = 20
    x = T.matrix('x')
    y = T.ivector('y')
    rng = numpy.random.RandomState(1234)
    # hidden layer
    hidden_layer_node = 500
    hidden_layer = HiddenLayer(rng, x, 0, 28*28, hidden_layer_node)
    # softmax as logistic regress
    soft_max = SoftMax(hidden_layer.y_given_x, y, hidden_layer_node, hidden_layer_node)
    # cost
    l2_lamda = 0.0001
    l2 = (hidden_layer.w ** 2).sum() + (soft_max.w ** 2).sum()
    cost = soft_max.cost + l2_lamda * l2
    # update params
    learning_rate = 0.01
    all_params = hidden_layer.params + soft_max.params
    g_all_params = [T.grad(cost, param) for param in all_params]
    updates = [
        (param, param - learning_rate * g_param) for param, g_param in zip(all_params, g_all_params)
    ]

    # train model
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # valid model
    validate_model = theano.function(
        inputs=[index],
        outputs=soft_max.error,
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # remember the best
    hidden_layer_best_w = hidden_layer.w.get_value(borrow=False)
    hidden_layer_best_b = hidden_layer.b.get_value(borrow=False)
    soft_max_best_w = soft_max.w.get_value(borrow=False)
    soft_max_best_b = soft_max.b.get_value(borrow=False)

    def finish_once(improve):
        if improve:
            hidden_layer_best_w = hidden_layer.w.get_value(borrow=False)
            hidden_layer_best_b = hidden_layer.b.get_value(borrow=False)
            soft_max_best_w = soft_max.w.get_value(borrow=False)
            soft_max_best_b = soft_max.b.get_value(borrow=False)
    sgd_train(train_set, valid_set, train_model, validate_model, finish_once, default_batch_size=batch_size)
    hidden_layer.w.set_value(hidden_layer_best_w)
    hidden_layer.b.set_value(hidden_layer_best_b)
    soft_max.w.set_value(soft_max_best_w)
    soft_max.b.set_value(soft_max_best_b)
