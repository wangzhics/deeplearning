import numpy
import theano
import theano.tensor as T
from theano_examples.new.model import SoftMax
from theano_examples.new.train import TrainModel


class LogisticModel(TrainModel):
    def __init__(self, train_set, valid_set):
        train_set_x, train_set_y = train_set
        valid_set_x, valid_set_y = valid_set
        soft_max = SoftMax(28*28, 10)
        g_w = T.grad(cost=soft_max.cost, wrt=soft_max.w)
        g_b = T.grad(cost=soft_max.cost, wrt=soft_max.b)
        learning_rate = 0.15
        updates = [(soft_max.w, soft_max.w - learning_rate * g_w),
                   (soft_max.b, soft_max.b - learning_rate * g_b)]
        index = T.lscalar()
        train_set_size = train_set.get_value(borrow=True).shape[0]
        batch_size = train_set_size // TrainModel.BATCH_COUNT
        self.train_model = theano.function(
            inputs=[index],
            outputs=soft_max.cost,
            updates=updates,
            givens={
                soft_max.x: train_set_x[index * batch_size: (index + 1) * batch_size],
                soft_max.y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )
        self.validate_model = theano.function(
            inputs=[index],
            outputs=soft_max.errors,
            givens={
                soft_max.x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                soft_max.y: valid_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        def train_batch(self, batch_index):
            return self.train_model

        def valid_error(self):
            validation_losses = [self.validate_model(i) for i in range(n_valid_batches)]
            return numpy.mean(validation_losses)

        def finish_once(self):
            pass

        def finish(self):
            pass

if __name__ == '__main__':
    soft_max = SoftMax(28*28, 10)
    g_w = T.grad(cost=soft_max.cost, wrt=soft_max.w)
    g_b = T.grad(cost=soft_max.cost, wrt=soft_max.b)
    learning_rate = 0.15
    updates = [(soft_max.w, soft_max.w - learning_rate * g_w),
               (soft_max.b, soft_max.b - learning_rate * g_b)]
    index = T.lscalar()
