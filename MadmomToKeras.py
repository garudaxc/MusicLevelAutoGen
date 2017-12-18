import madmom
from madmom.ml.nn import NeuralNetworkEnsemble
from madmom.models import DOWNBEATS_BLSTM
import pickle
import tensorflow as tf
import numpy as np
import os


class TensorLayer():
    def __init__(self, tensor, stride):
        self.tensor = tensor
        self.stride = stride


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
    
    w = tf.concat([ti, tforget, tc, to], 1, name='lstm_layer')

    stride = layer.input_gate.bias.shape[0]
    l = TensorLayer(w, stride)

    return l

# def ConvertForwardLayer(layer):
#     assert type(layer) == madmom.ml.nn.layers.FeedForwardLayer



def ConvertBidirectionLayer(layer):
    assert type(layer) == madmom.ml.nn.layers.BidirectionalLayer

    fwd_tensor = ConvertLayerWeight(layer.fwd_layer)
    back_tensor = ConvertLayerWeight(layer.bwd_layer)

    return [fwd_tensor, back_tensor]

def ConvertMultiLayerNetwrok(network):
    assert type(network) == madmom.ml.nn.NeuralNetwork

    layers = network.layers
    newLayers = []
    newLayers.append(ConvertBidirectionLayer(layers[0]))
    newLayers.append(ConvertBidirectionLayer(layers[1]))
    newLayers.append(ConvertBidirectionLayer(layers[2]))
    return newLayers

def RunMultiLayerNetwrok(network, data, data_size):
    for layer in network:
        data = RunBidirectionLayer(layer, data, data_size)

    return data



def RunLayer(tensorLayer, data, data_size):

    tensor = tensorLayer.tensor
    nStride = tensorLayer.stride

    ig = tensor[:,0*nStride:1*nStride]
    fg = tensor[:,1*nStride:2*nStride]
    cg = tensor[:,2*nStride:3*nStride]
    og = tensor[:,3*nStride:4*nStride]

    data_type = tensor.dtype
    prev = tf.zeros([nStride], dtype=data_type)
    state = tf.zeros([nStride], dtype=data_type)
    # print(prev.shape)
    # print(state.shape)
    one = tf.constant([1.0], dtype=data_type)

    count = data_size
    
    out = tf.TensorArray(dtype = data_type, size=count)

    cond = lambda i, *_:tf.less(i, count)

    def body(_i, _prev, _state, _out):
        dd = data[_i]

        v = tf.concat([dd, _prev, _state, one], 0)
        l = tf.shape(v)
        v = tf.reshape(v, (1, l[0]))

        i = tf.sigmoid(tf.matmul(v, ig))
        f = tf.sigmoid(tf.matmul(v, fg))
        c = tf.tanh(tf.matmul(v, cg))
        newState = c * i + _state * f

        v = tf.concat([dd, _prev, newState[0], one], 0)

        v = tf.reshape(v, (1, l[0]))
        o = tf.sigmoid(tf.matmul(v, og))

        newPrev = tf.tanh(newState) * o
        newOut = _out.write(_i, newPrev[0])

        return tf.add(_i, 1), newPrev[0], newState[0], newOut

    ii = tf.constant(0, dtype=tf.int32)
    a, b, c, d = tf.while_loop(cond, body, [ii, prev, state, out])
    d = d.stack()

    return d


def RunBidirectionLayer(tensors, input_data, data_size):
    data_type = tensors[0].tensor.dtype

    data0 = input_data
    data1 = tf.reverse(data0, [0])
    res0 = RunLayer(tensors[0], data0, data_size)
    res1 = RunLayer(tensors[1], data1, data_size)
    res1 = tf.reverse(res1, [0])

    result = tf.concat([res0, res1], axis=1)
    
    return result


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

    data = tf.Variable(input_data)

    out = RunLayer(w, data)
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
        input_data = input_data[:1000]
    
    print('file loaded', type(input_data), input_data.shape, input_data.dtype)
    data_type = layer.input_gate.weights.dtype
    
    t = time.time()
    result = layer.activate(input_data)

    print(result[-1], time.time() - t)
    print('aaa')

    w = ConvertLayerWeight(layer)

    data = tf.Variable(input_data)
    out = RunLayer(w, data, len(input_data))
    with tf.Session() as sess:        
        sess.run(tf.global_variables_initializer())

        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()      
        
        t = time.time()        
        r = sess.run(out, options=options, run_metadata=run_metadata)
        print(r[-1], time.time() - t)    

        from tensorflow.python.client import timeline
        fetched_timeline = timeline.Timeline(run_metadata.step_stats, graph=sess.graph)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        with open('timeline_01.json', 'w') as f:
            f.write(chrome_trace)



def Test4(layer):
    input_size = 314
    node_size = 25

    import pickle
    import time

    fillename = 'd:/work/signal_data.pk'
    if os.name != 'nt':
        fillename = '/Users/xuchao/Documents/work/signal_data.pk'
    with open(fillename, 'rb') as file:
        input_data = pickle.load(file)
    
    print('file loaded', type(input_data), input_data.shape, input_data.dtype)
    
    t = time.time()
    result = layer.activate(input_data)

    print(result[-1], time.time() - t)
    print('aaa')

    w = ConvertBidirectionLayer(layer)
    data = tf.Variable(input_data)
    out = RunBidirectionLayer(w, data, len(input_data))
    with tf.Session() as sess:        
        sess.run(tf.global_variables_initializer())
        t = time.time()
        r = sess.run(out)
        print(r[-1], time.time() - t)

def Test5(network):
    input_size = 314
    node_size = 25

    import pickle
    import time

    fillename = 'd:/work/signal_data.pk'
    if os.name != 'nt':
        fillename = '/Users/xuchao/Documents/work/signal_data.pk'
    with open(fillename, 'rb') as file:
        input_data = pickle.load(file)
    
    print('file loaded', type(input_data), input_data.shape, input_data.dtype)
    
    t = time.time()
    result = network.process(input_data)

    print(result[-1], time.time() - t)
    print('aaa')

    w = ConvertMultiLayerNetwrok(network)
    data = tf.Variable(input_data, name='input_data')
    out = RunMultiLayerNetwrok(w, data, len(input_data))

    with tf.Session() as sess:        
        sess.run(tf.global_variables_initializer())

        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()      
        
        t = time.time()        
        r = sess.run(out, options=options, run_metadata=run_metadata)
        print(r[-1], time.time() - t)

        train_writer = tf.summary.FileWriter('d:/work/train', sess.graph)
        # train_writer.add_graph(g)
        # train_writer.add_summary(summary)
        train_writer.close()

        from tensorflow.python.client import timeline
        fetched_timeline = timeline.Timeline(run_metadata.step_stats, graph=sess.graph)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        with open('timeline_01.json', 'w') as f:
            f.write(chrome_trace)



def PrintLayerParam(layer):
    assert type(layer) == madmom.ml.nn.layers.BidirectionalLayer
    
    lstmLayer = layer.fwd_layer
    assert type(lstmLayer) == madmom.ml.nn.layers.LSTMLayer

    gate = lstmLayer.input_gate
    print('weights shape', gate.weights.shape)

def PrintFwdLayer(layer):
    assert type(layer) == madmom.ml.nn.layers.FeedForwardLayer

    print('weights', layer.weights.shape)
    print('bais', layer.bias.shape)
    print('activation func', layer.activation_fn)



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
    PrintFwdLayer(BiLayers[3])
    print(type(BiLayers))
    # print(BiLayers[3])  # a feedForwardLayer
    BiLayers.pop()
    # return

    layer = BiLayers[0]
    assert type(layer) == madmom.ml.nn.layers.BidirectionalLayer


    lstmLayer = layer.fwd_layer
    assert type(lstmLayer) == madmom.ml.nn.layers.LSTMLayer

    # Test1(lstmLayer.input_gate)
    # Test2(lstmLayer)
    Test3(lstmLayer)
    # Test4(layer)
    # Test5(network)

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

    # TFTest()
    Do()
    # TensorOper()
    # TFRnn()
        
