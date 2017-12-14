import madmom
from madmom.ml.nn import NeuralNetworkEnsemble
from madmom.models import DOWNBEATS_BLSTM
import pickle
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

from tensorflow.python.eager import context
from tensorflow.python.estimator import util as estimator_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import variable_scope as vs


class TestClass():

    z = 111

    def foo(self):
        self.x = 10
        self.y = 20

    def __init__(self):
        self.bbb = 'asd'

def GateParameter(gate):
    assert type(gate) == madmom.ml.nn.layers.Gate or type(gate) == madmom.ml.nn.layers.Cell
    # assert gate.peephole_weights is not None

    # print(type(gate))
    # if gate.peephole_weights is not None:
    #     print('peephole', gate.peephole_weights.shape)
    # else:
    #     print('peephole is none')
    
    # print('recurrent weight', gate.recurrent_weights.shape)
    # print('weights', gate.weights.shape)
    # print('bias', gate.bias.shape)
    # print('activite', gate.activation_fn)    
    data_type = gate.bias.dtype
    if gate.peephole_weights is not None:
        peephole_mat = np.diagflat(gate.peephole_weights).astype(data_type)
    else:
        peephole_mat = np.zeros_like(gate.recurrent_weights, dtype=data_type)
    # peephole_mat = np.diagflat(gate.peephole_weights)
    mat = np.vstack((gate.weights, gate.recurrent_weights, peephole_mat, gate.bias))
    return mat

def ConvertLayerWeight(layer):
    mi = GateParameter(layer.input_gate)
    mf = GateParameter(layer.forget_gate)
    mc = GateParameter(layer.cell)
    mo = GateParameter(layer.output_gate)

    ti = tf.Variable(mi)
    tforget = tf.Variable(mf)
    tc = tf.Variable(mc)
    to = tf.Variable(mo)
    
    w = tf.concat([ti, tforget, tc, to], 1)
    return w

def RunLayer(layerTensor, nStride, input_data):
    ig = layerTensor[:,0*nStride:1*nStride]
    fg = layerTensor[:,1*nStride:2*nStride]
    cg = layerTensor[:,2*nStride:3*nStride]
    og = layerTensor[:,3*nStride:4*nStride]

    data_type = layerTensor.dtype
    data = tf.Variable(input_data)
    prev = tf.zeros([nStride], dtype=data_type)
    state = tf.zeros([nStride], dtype=data_type)
    # print(prev.shape)
    # print(state.shape)
    one = tf.constant([1.0], dtype=data_type)

    count = data.shape[0]
    # count = tf.constant(count)
    # out = tf.zeros([count, nStride], dtype=data_type)
    out = tf.TensorArray(dtype = data_type, size=count)

    cond = lambda i, *_:tf.less(i, count)

    def body(_i, _prev, _state, _out):
        dd = data[_i]

        v = tf.concat([dd, _prev, _state, one], 0)
        v = tf.reshape(v, (1, v.shape[0]))

        i = tf.sigmoid(tf.matmul(v, ig))
        f = tf.sigmoid(tf.matmul(v, fg))
        c = tf.tanh(tf.matmul(v, cg))
        newState = c * i + _state * f

        v = tf.concat([dd, _prev, newState[0], one], 0)

        v = tf.reshape(v, (1, v.shape[0]))
        o = tf.sigmoid(tf.matmul(v, og))

        newPrev = tf.tanh(newState) * o
        newOut = _out.write(_i, newPrev[0])

        return tf.add(_i, 1), newPrev[0], newState[0], newOut

    ii = tf.constant(0, dtype=tf.int32)
    a, b, c, d = tf.while_loop(cond, body, [ii, prev, state, out])
    d = d.stack()

    return d


def Test1(gate):

    input_size = 314
    node_size = 25
    
    # print(gate.peephole_weights.dtype)
    data_type = gate.peephole_weights.dtype
    input_data = np.random.randn(input_size).astype(data_type)
    # print(data_type)
    # return
    prv_data = np.random.randn(node_size).astype(data_type)
    state_data = np.random.randn(node_size).astype(data_type)

    result = gate.activate(input_data, prv_data, state_data)

    weight_mat = GateParameter(gate)
    print(result)
    print('aaa')
    # result2 = gate.activate(input_data, prv_data, state_data)
    # print(result2)
    # return

    v0 = tf.Variable(weight_mat)
    v1 = tf.Variable(input_data)
    v2 = tf.Variable(prv_data)
    v3 = tf.Variable(state_data)
    v4 = tf.constant([1.0])
    v_input = tf.concat([v1, v2, v3, v4], 0)
    v_input = tf.reshape(v_input, (1, v_input.shape[0]))

    result2 = tf.matmul(v_input, v0)
    result2 = tf.sigmoid(result2)

    with tf.Session() as sess:        
        sess.run(tf.global_variables_initializer())
        r = sess.run(result2)
        print(r)


def Test2(layer):
    input_size = 314
    node_size = 25

    data_type = layer.input_gate.peephole_weights.dtype
    input_data = np.random.randn(2, input_size).astype(data_type)

    result = layer.activate(input_data)
    print(result)
    print('aaa')

    w = ConvertLayerWeight(layer)
    out = RunLayer(w, node_size, input_data)
    with tf.Session() as sess:        
        sess.run(tf.global_variables_initializer())
        r = sess.run(out)
        print(r)


def Test3(layer):
    input_size = 314
    node_size = 25

    import pickle
    import time

    with open('d:/work/signal_data.pk', 'rb') as file:
        input_data = pickle.load(file)
    
    print('file loaded', type(input_data), input_data.shape, input_data.dtype)
    data_type = layer.input_gate.weights.dtype
    
    t = time.time()
    result = layer.activate(input_data)

    print(result[-1], time.time() - t)
    print('aaa')

    w = ConvertLayerWeight(layer)
    out = RunLayer(w, node_size, input_data)
    with tf.Session() as sess:        
        sess.run(tf.global_variables_initializer())
        t = time.time()
        r = sess.run(out)
        print(r[-1], time.time() - t)


def PrintLayerParam(layer):
    assert type(layer) == madmom.ml.nn.layers.BidirectionalLayer
    
    lstmLayer = layer.fwd_layer
    assert type(lstmLayer) == madmom.ml.nn.layers.LSTMLayer

    gate = lstmLayer.input_gate
    print('weights shape', gate.weights.shape)




def Do():    
    nn = NeuralNetworkEnsemble.load(DOWNBEATS_BLSTM)

    assert len(nn.processors) == 2
    assert type(nn.processors[0]) == madmom.processors.ParallelProcessor
    assert nn.processors[1] == madmom.ml.nn.average_predictions

    # 8个相同的网络
    pp = nn.processors[0]
    # for p in pp.processors:
    #     print(p)

    # 每个网络是个三层双向网络
    network = pp.processors[0]
    assert type(network) == madmom.ml.nn.NeuralNetwork

    BiLayers = network.layers
    print('biLayers size', len(BiLayers))
    PrintLayerParam(BiLayers[0])
    PrintLayerParam(BiLayers[1])
    PrintLayerParam(BiLayers[2])
    # print(BiLayers[3])  # a feedForwardLayer
    return

    layer = BiLayers[0]
    assert type(layer) == madmom.ml.nn.layers.BidirectionalLayer


    lstmLayer = layer.fwd_layer
    assert type(lstmLayer) == madmom.ml.nn.layers.LSTMLayer

    # Test1(lstmLayer.input_gate)
    # Test2(lstmLayer)
    Test3(lstmLayer)

    # GateParameter(lstmLayer.cell)
    # GateParameter(lstmLayer.input_gate)
    # GateParameter(lstmLayer.forget_gate)
    # GateParameter(lstmLayer.output_gate)
   
    print(type(lstmLayer.cell))
    print(lstmLayer.activation_fn)
    

    # pickle.dump(nn, open('d:/output/test.txt', 'wb'), protocol=0)
    #  for i in nn.processors:
    #     print(i)


if __name__ == '__main__':

    if True:
        # TFTest()
        Do()
        # TensorOper()
        # TFRnn()
        

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