import theano.tensor as T


class CrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, inputs, labels):
        # Your codes here
        return T.sum(T.nnet.categorical_crossentropy(inputs,labels))

