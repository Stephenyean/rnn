from layers import RNN, LSTM, Softmax, Linear
from loss import CrossEntropyLoss
from optimizer import *
from network import Network
from data_preparation import load_data
from solve_rnn import solve_rnn

import theano.tensor as T

X_train, y_train, X_test, y_test = load_data()

HIDDEN_DIM = 128
INPUT_DIM = 20
OUTPUT_DIM = 10

model = Network()
model.add(RNN('rnn1', HIDDEN_DIM, INPUT_DIM, 0.1))      # output shape: 4 x HIDDEN_DIM
model.add(Linear('fc', HIDDEN_DIM, OUTPUT_DIM, 0.1))    # output shape: 4 x OUTPUT_DIM
model.add(Softmax('softmax'))

loss = CrossEntropyLoss('xent')

#optim = SGDOptimizer(0.01, 0, 0)
optim = RMSpropOptimizer(0.001)
input_placeholder = T.fmatrix('input')
label_placeholder = T.fmatrix('label')

model.compile(input_placeholder, label_placeholder, loss, optim)

MAX_EPOCH = 10
DISP_FREQ = 1000
TEST_FREQ = 10000

solve_rnn(model, X_train, y_train, X_test, y_test,
          MAX_EPOCH, DISP_FREQ, TEST_FREQ)
