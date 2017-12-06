import madmom
from madmom.ml.nn import NeuralNetworkEnsemble
from madmom.models import DOWNBEATS_BLSTM
import pickle
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn


class TestClass():

    z = 111

    def foo(self):
        self.x = 10
        self.y = 20

    def __init__(self):
        self.bbb = 'asd'

def GateParameter(gate):
    assert type(gate) == madmom.ml.nn.layers.Gate
    if gate.peephole_weights is not None:
        print('peephole', gate.peephole_weights.shape)
    
    print('recurrent weight', gate.recurrent_weights.shape)
    print('weights', gate.weights.shape)
    print('bias', gate.bias.shape)


# def KerasLayers():
#     # layer = keras.layers.Dense(32)
#     layer = keras.layers.LSTM(26, return_sequences=False, stateful=False)

#     # layer.build(input_shape=(16, 314))
#     layer.build(input_shape=(16, 1, 314))

#     w = layer.weights
#     print(len(w))
#     print(type(w[0]))

#     weights = layer.get_weights()
#     for w in weights:
#         print(w.shape)
#     # print(len(weights))
#     # print(weights)
#     # print(layer.count_params())

def TFRnn():
    fwCell = rnn.LSTMCell(26, True)
    print(fwCell)


    #lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)


def TensorOper():
    import time

    a = np.random.randn(100, 1000, 10)
    
    t = time.time()
    b = a * 4.5 
    print('t0', time.time() - t)
    
    node1 = tf.constant(a, dtype=tf.float32)
    node2 = tf.constant(4.5) # also tf.float32 implicitly
    c = tf.multiply(node1, node2)
    print(type(node1))
    print(c)

    sess = tf.Session()
    
    t = time.time()
    d = sess.run(c)
    print('t1', time.time() - t)
    # print(d)


def Do():    
    nn = NeuralNetworkEnsemble.load(DOWNBEATS_BLSTM)

    assert len(nn.processors) == 2
    assert type(nn.processors[0]) == madmom.processors.ParallelProcessor
    assert nn.processors[1] == madmom.ml.nn.average_predictions

    pp = nn.processors[0]
    # for p in pp.processors:
    #     print(p)

    network = pp.processors[0]
    assert type(network) == madmom.ml.nn.NeuralNetwork

    BiLayers = network.layers
    layer = BiLayers[0]
    assert type(layer) == madmom.ml.nn.layers.BidirectionalLayer

    lstmLayer = layer.fwd_layer
    assert type(lstmLayer) == madmom.ml.nn.layers.LSTMLayer


    GateParameter(lstmLayer.input_gate)
    GateParameter(lstmLayer.forget_gate)
    GateParameter(lstmLayer.output_gate)
   
    print(type(lstmLayer.cell))
    print(lstmLayer.activation_fn)
    

    # pickle.dump(nn, open('d:/output/test.txt', 'wb'), protocol=0)
    #  for i in nn.processors:
    #     print(i)


if __name__ == '__main__':

    if True:
        # Do()
        # TensorOper()
        TFRnn()
        

    if False:
        a = TestClass()
        c = TestClass()
        c.z = 12

        c.foo()

        print(TestClass.__dict__)    
        print(TestClass.__name__)


        print(type(c.foo))

        # print(c.__str__)
        # print(c.__class__)

        # print(dir(c))