import pickle
import gzip
import matplotlib.pyplot as plt

with gzip.open('../mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f,encoding='latin1')
    train_x, train_y = train_set
    image_data = train_x[1].reshape((28, 28))
    print(image_data)
    plt.imsave('0.png', image_data)