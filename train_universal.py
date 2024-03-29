from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

import numpy as np
import os
data_path = '/temp'
os.makedirs('./outputs', exist_ok=True)

try:    
    from azureml.core.run import Run
    run = Run.get_context()
except:
    run = None

print('fetching MNIST data...')
mnist = fetch_openml('mnist_784',as_frame=False)
mnist['target'] = np.array([int(x) for x in mnist['target']])

print(mnist)

# use a random subset of n records to reduce training time.
n = 20000
shuffle_index = np.random.permutation(70000)[:n]
X, y = mnist['data'][shuffle_index], mnist['target'][shuffle_index]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

lr = LogisticRegression()
print("training a logistic regression model...")
lr.fit(X_train, y_train)
print(lr)
y_hat = lr.predict(X_test)
acc = np.average(np.int32(y_hat == y_test))

print('Overall accuracy:', acc)

if run is not None:
    run.log('accuracy', acc)
