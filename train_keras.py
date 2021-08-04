import argparse
import json
import os
from azureml.core.model import Model
from azureml.core import Run
import pickle
from tensorflow import keras
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.callbacks import Callback
import numpy as np

parser = argparse.ArgumentParser(description='MNIST Train')
parser.add_argument('--data_path', type=str, dest='data_path', help='data folder mounting point')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--hidden', type=int, default=100)
parser.add_argument('--dropout', type=float)

args = parser.parse_args()

run = Run.get_context()

mnist_fn = 'dataset/mnist.pkl' if args.data_path is None else args.data_path
with open(mnist_fn,'rb') as f:
    X,y = pickle.load(f)

X /= 255.0
y = keras.utils.to_categorical(y,10)

n = int(0.8*X.shape[0])
x_train = X[0:n]
y_train = y[0:n]
x_test = X[n:]
y_test = y[n:]

model = keras.models.Sequential()
model.add(Dense(args.hidden,input_shape=(784,),activation='relu'))
if args.dropout is not None and args.dropout<1:
    model.add(Dropout(args.dropout))
model.add(Dense(10,activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

run = Run.get_context()

class LogRunMetrics(Callback):
    # callback at the end of every epoch
    def on_epoch_end(self, epoch, log):
        # log a value repeated which creates a list
        run.log("Loss", log["loss"])
        run.log("Accuracy", log["accuracy"])


model.fit(x_train, y_train,
          batch_size=args.batch_size,
          epochs=args.epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[LogRunMetrics()])

score = model.evaluate(x_test, y_test, verbose=0)
loss = score[0]
acc = score[1]
print('Test loss:', loss)
print('Test accuracy:', acc)

# Log metrics
run.log("Test Loss", loss)
run.log('Test Accuracy', np.float(acc))


# Save the model
os.makedirs('outputs',exist_ok=True)
model.save('outputs/mnist_model.hdf5')

