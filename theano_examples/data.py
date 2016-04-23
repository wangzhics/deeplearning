import pickle
import gzip


def load_data():
    with gzip.open('../mnist.pkl.gz', 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
            return train_set, valid_set, test_set