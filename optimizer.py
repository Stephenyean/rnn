import theano.tensor as T
from utils import sharedX


class SGDOptimizer(object):
    def __init__(self, learning_rate, weight_decay, momentum):
        self.lr = learning_rate
        self.wd = weight_decay
        self.mm = momentum

    def get_updates(self, cost, params):
        grads = T.grad(cost=cost, wrt=params)
        updates = []
        for p, g in zip(params, grads):
            d = sharedX(p.get_value() * 0.0)
            new_d = self.mm * d - self.lr * (g + self.wd * p)
            updates.append((d, new_d))
            updates.append((p, p + new_d))

        return updates


class RMSpropOptimizer(object):
    def __init__(self, learning_rate, rho=0.9, eps=1e-8):
        # rho: decay_rate
        self.lr = learning_rate
        self.rho = rho
        self.eps = eps

    def get_updates(self, cost, params):
        # Your codes here

