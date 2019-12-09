from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import os
import pickle
import numpy as np

print('fetching MNIST data...')
mnist = fetch_openml('mnist_784')
mnist['target'] = np.array([int(x) for x in mnist['target']])

# use a random subset of n records to reduce training time.
n = 20000
shuffle_index = np.random.permutation(70000)[:n]
X, y = mnist['data'][shuffle_index], mnist['target'][shuffle_index]

os.makedirs('dataset')
with open('dataset/mnist.pkl','wb') as f:
    pickle.dump((X,y),f)

print('Done')
