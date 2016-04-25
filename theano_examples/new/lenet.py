import numpy
import theano
import theano.tensor as T
from theano_examples.new.train import load_data, sgd_train
from theano_examples.new.model import SoftMax, HiddenLayer, LeNetConvPoolLayer

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
    rng = numpy.random.RandomState(23455)
    # layer 1: input: (20, 1, 28, 28) by filter (2, 1, 5, 5) -> (20, 2, 24, 24) by pool (2, 2) -> (20, 2, 12, 12)
    layer1_filter_count = 2
    layer1_input_shape = (batch_size, 1, 28, 28)
    layer1_filter_shape = (layer1_filter_count, 1, 5, 5)
    layer1_input = x.reshape((batch_size, 1, 28, 28))
    layer1 = LeNetConvPoolLayer(rng, layer1_input, layer1_input_shape, layer1_filter_shape)
    # layer 2: input: (20, 2, 12, 12) by filter (2, 2, 5, 5) -> (20, 4, 8, 8) by pool (2, 2) -> (20, 4, 4, 4)
    layer2_filter_count = 4
    layer2_input_shape = (batch_size, layer1_filter_count, 12, 12)
    layer2_filter_shape = (layer2_filter_count, layer1_filter_count, 5, 5)
    layer2 = LeNetConvPoolLayer(rng, layer1.output, layer2_input_shape, layer2_filter_shape)
    # hidden layer: (20, 4, 4, 4) -> 20 * (4 * 4 * 4)
    hidden_layer_node = 50
    hidden_layer_input = layer2.output.flatten(2)
    hidden_layer = HiddenLayer(rng, hidden_layer_input, 0, 4*4*4, hidden_layer_node)
    # softmax as logistic regress
    soft_max = SoftMax(hidden_layer.y_given_x, y, hidden_layer_node, 10)
    # train
    learning_rate = 0.01
    all_params = layer1.params + layer2.params + hidden_layer.params + soft_max.params
    g_all_params = [T.grad(soft_max.cost, param) for param in all_params]
    updates = [
        (param, param - learning_rate * g_param) for param, g_param in zip(all_params, g_all_params)
    ]
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
    # valid model
    validate_model = theano.function(
        inputs=[index],
        outputs=soft_max.error,
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    def finish_once(improve):
        print("improve is ", improve)
    sgd_train(train_set, valid_set, train_model, validate_model, finish_once, default_batch_size=batch_size)


