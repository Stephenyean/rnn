import theano
import theano.tensor as T
import numpy as np
from utils import sharedX


class Layer(object):
    def __init__(self, name, trainable=False):
        self.name = name
        self.trainable = trainable

    def forward(self, inputs):
        pass

    def params(self):
        pass


class Relu(Layer):
    def __init__(self, name):
        super(Relu, self).__init__(name)

    def forward(self, inputs):
        # Your codes here
        return T.nnet.relu(inputs)


class Softmax(Layer):
    def __init__(self, name):
        super(Softmax, self).__init__(name)

    def forward(self, inputs):
        # Your codes here
        return T.nnet.softmax(inputs)

class Linear(Layer):
    def __init__(self, name, inputs_dim, num_output, init_std):
        super(Linear, self).__init__(name, trainable=True)
        self.W = sharedX(np.random.randn(inputs_dim, num_output) * init_std, name=name + '/W')
        self.b = sharedX(np.zeros((num_output)), name=name + '/b')

    def forward(self, inputs):
        # Your codes here
        return T.dot(T.flatten(inputs,2),self.W) + self.b

    def params(self):
        return [self.W, self.b]


class RNN(Layer):
    def __init__(self, name, hidden_dim, input_dim, init_std):
        super(RNN, self).__init__(name, trainable=True)
        self.hidden_dim = hidden_dim
        self.Wx = sharedX(np.random.randn(input_dim, hidden_dim) * init_std, name=name + '/Wx')
        self.Wh = sharedX(np.random.randn(hidden_dim, hidden_dim) * init_std, name=name + '/Wh')
        self.b = sharedX(np.zeros((hidden_dim)), name=name + '/b')

    def forward(self, inputs):
        """
        Args:
            inputs: a sequence size of 4 x 20, stands for two 4-digital numbers

        Returns:
            hs:
                hs[0]: a sequence size of 4 x hidden_dim,
                       stands for step function output at each time
                hs[1]: a sequence size of 4 x hidden_dim,
                       stands for hidden states at each timestamp
        Notice:
            In this simple RNN, step function output is same as hidden states
            But for LSTM, since hidden states also include cell, hs[0] and hs[1:] are different
        """
        hs, _ = theano.scan(
            self.step,          # forward input to output with step function for each time
            sequences=inputs,
            outputs_info=[None, T.zeros(self.hidden_dim)])  # initial values for output and hidden states

        return hs[0]

    def step(self, x_t, h_t_prev):
        """
        Args:
            x_t: input at current timestamp
            h_t_prev: hidden states at previous timestamp

        Returns:
            [h_t, h_t]: output value and updated hidden states value
        """
        h_t = T.tanh(T.dot(x_t, self.Wx) + T.dot(h_t_prev, self.Wh) + self.b)
        return [h_t, h_t]

    def params(self):
        return [self.Wx, self.Wh, self.b]


class LSTM(Layer):
    def __init__(self, name, hidden_dim, input_dim, init_std):
        super(LSTM, self).__init__(name, trainable=True)
        self.hidden_dim = hidden_dim
        self.Wi = sharedX(np.random.randn(input_dim, hidden_dim) * init_std, name=name + '/Wi')
        self.Ui = sharedX(np.random.randn(hidden_dim, hidden_dim) * init_std, name=name + '/Ui')
        self.Vi = sharedX(np.random.randn(hidden_dim, hidden_dim) * init_std, name=name + '/Vi')
        self.b_i = sharedX(np.random.randn(hidden_dim) * init_std, name = name+ '/bi')
        self.Wf = sharedX(np.random.randn(input_dim, hidden_dim) * init_std, name=name + '/Wf')
        self.Uf = sharedX(np.random.randn(hidden_dim, hidden_dim) * init_std, name=name + '/Uf')
        self.Vf = sharedX(np.random.randn(hidden_dim, hidden_dim) * init_std, name=name + '/Vf')
        self.b_f = sharedX(np.random.randn(hidden_dim) * init_std, name = name+ '/bf')
        self.Wo = sharedX(np.random.randn(input_dim, hidden_dim) * init_std, name=name + '/Wo')
        self.Uo = sharedX(np.random.randn(hidden_dim, hidden_dim) * init_std, name=name + '/Uo')
        self.Vo = sharedX(np.random.randn(hidden_dim, hidden_dim) * init_std, name=name + '/Vo')
        self.b_o = sharedX(np.random.randn(hidden_dim) * init_std, name = name+ '/bo')
        self.Wc = sharedX(np.random.randn(input_dim, hidden_dim) * init_std, name=name + '/Wc')
        self.Uc = sharedX(np.random.randn(hidden_dim, hidden_dim) * init_std, name=name + '/Uc')
        self.b_c = sharedX(np.random.randn(hidden_dim) * init_std, name = name+ '/bc')
        # Your codes here, do weights initilization

    def forward(self, inputs):
        results, _ = theano.scan(
            self.step,
            sequences=inputs,
            outputs_info=[None, T.zeros(self.hidden_dim), T.zeros(self.hidden_dim)])

        return results[0]

    def step(self, x_t, h_t_prev, c_t_prev):
        """
        Args:
            x_t: input at current timestamp
            h_t_prev: hidden states at previous timestamp
            c_t_prev: cell states at previous timestamp
        """
        # Your codes here
        i_t = T.nnet.nnet.sigmoid(T.dot(x_t, self.Wi) + T.dot(h_t_prev,self.Ui) + T.dot(c_t_prev, self.Vi) + self.b_i)
        f_t = T.nnet.nnet.sigmoid(T.dot(x_t, self.Wf) + T.dot(h_t_prev,self.Uf) + T.dot(c_t_prev, self.Vf) + self.b_f)
        o_t = T.nnet.nnet.sigmoid(T.dot(x_t, self.Wo) + T.dot(h_t_prev,self.Uo) + T.dot(c_t_prev, self.Vo) + self.b_o)
        _c_t = T.tanh(T.dot(x_t, self.Wc) + T.dot(h_t_prev, self.Uc) +  self.b_c)
        #c_t1,update1 = theano.scan(lambda x,y: T.dot(x,y), sequences=[f_t, c_t_prev])
	#c_t2,update2 = theano.scan(lambda x,y: T.dot(x,y), sequences=[i_t,_c_t])
        #c_t = c_t1 + c_t2
	#h_t = theano.scan(lambda x,y: T.dot(x,y), sequences=[o_t, T.tanh(c_t)])
        c_t = f_t * c_t_prev + i_t * _c_t
	h_t = o_t * T.tanh(c_t)
	return [h_t, h_t, c_t]
    
    def params(self):
        return [self.Wi, self.Ui, self.Vi, self.b_i, self.Wf, self.Uf, self.Vf, self.b_f, self.Wo, self.Uo, self.Vo, self.b_o, self.Wc, self.Uc, self.b_c]
        # Your codes here
