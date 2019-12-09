import argparse
import json
import os
from azureml.core import Run
from azureml.core.model import Model
import pickle
import keras
from keras.layers import Dense,Dropout

parser = argparse.ArgumentParser(description='MNIST Train')
parser.add_argument('--data_folder', type=str, dest='data_folder', help='data folder mounting point')
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--hidden', type=int, default=100)
parser.add_argument('--dropout', type=float)

args = parser.parse_args()

mnist_fn = 'dataset/mnist.pkl' if args.data_folder is None else os.path.join(args.data_folder, 'mnist_data','mnist.pkl')
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
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=args.batch_size,
          epochs=args.epochs,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

os.makedirs('outputs',exist_ok=True)
model.save('outputs/mnist_model.hdf5')

# Log metrics
try:
    run = Run.get_context()
    run.log('Test Loss', score[0])
    run.log('Accuracy', score[1])
except:
    print("Running locally")
