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
    with gzip.open('../mnist.pkl.gz', 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
            return shared_dataset(train_set), shared_dataset(valid_set), shared_dataset(test_set)


class TrainModel:

    BATCH_COUNT = 100

    def __init__(self, model_size):
        self.batch_count = model_size // 100

    def train_batch(self, batch_index):
        pass

    def valid_error(self):
        pass

    def finish_once(self):
        pass

    def finish(self):
        pass

    def train_one(self):
        for i in range(TrainModel.BATCH_COUNT):
            self.train_batch(i)

    def get_batch_count(self, set_size):
        return set_size // TrainModel.BATCH_COUNT

    def run(self):
        running = True
        # early stop
        not_improve = 0
        i = 0
        while running:
            self.train_one()
            valid_error_rate = self.valid_error()
            if valid_error_rate < best_valid_error_rate:
                best_valid_error_rate = valid_error_rate
                self.save_params()
                not_improve = 0
            else:
                not_improve += 1
            if not_improve > 5:
                running = False