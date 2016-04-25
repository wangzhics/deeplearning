import time
import pickle
import gzip
import numpy
import theano
import theano.tensor as T


def shared_dataset(data_xy, borrow=True):
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
    return shared_x, T.cast(shared_y, 'int32')


def load_data():
    with gzip.open('../../mnist.pkl.gz', 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
            return shared_dataset(train_set), shared_dataset(valid_set), shared_dataset(test_set)


def sgd_train(train_set, valid_set, train_model, valid_model, finish_once, default_batch_size=600):
    train_set_x, train_set_y = train_set
    valid_set_x, valid_set_y = valid_set
    train_set_size = train_set_x.get_value(borrow=True).shape[0]
    valid_set_size = valid_set_x.get_value(borrow=True).shape[0]
    train_set_batch = train_set_size // default_batch_size
    valid_set_batch = valid_set_size // default_batch_size
    # sgd train
    running = True
    best_valid_error_rate = 1.0
    # early stop
    not_improve = 0
    i = 1
    while running:
        print("Train %d Begin" % i)
        start = time.clock()
        _train_one(train_model, train_set_batch)
        end = time.clock()
        print("Train %d Finish: Spend %f s" % (i, end-start))
        valid_error_rate = _valid_error(valid_model, valid_set_batch)
        if valid_error_rate < best_valid_error_rate:
            best_valid_error_rate = valid_error_rate
            finish_once(True)
            not_improve = 0
        else:
            finish_once(False)
            not_improve += 1
        if not_improve > 5:
            running = False
        # debug info
        print("    Valid Error Rate %f%%" % (valid_error_rate*100))
        i += 1


def _train_one(train_model, train_count):
        for i in range(train_count):
            train_model(i)


def _valid_error(valid_model, batch_count):
    validation_losses = [valid_model(i) for i in range(batch_count)]
    return numpy.mean(validation_losses)
