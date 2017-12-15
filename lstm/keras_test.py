
import tensorflow as tf



# def KerasLayers():
#     import keras

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

def TensorArrayTest():
    # array = tf.TensorArray(dtype = tf.float32, size=2)
    # array = array.write(0, [1.2, 2.0])
    # array = array.write(1, [1.0, 2.0])
    # b = array.stack()

    
    v1 = tf.Variable([[4, 9, 0, 3, 6], [0, 0, 0, 0, 1]], dtype=tf.float32)
    array = tf.TensorArray(dtype=v1.dtype, handle=v1)
    b = array.read(0)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    y = sess.run(b)
    print(y)



def LoopTest():
    v0 = tf.Variable([4, 9, 0, 3, 6], dtype=tf.float32)
    v1 = tf.Variable([[4, 9, 0, 3, 6], [0, 0, 0, 0, 1]], dtype=tf.float32)

    count = 2
    cond = lambda i, v : tf.less(i, 2)

    def body(i, v):
        # a = v[i:i+1]
        # b = v[i+1:i+2]
        # c = tf.assign(b, a)
        v2 = v + v1[i]
        v1[i] = v2
        return tf.add(i, 1), v2

    ii = tf.constant(0, dtype=tf.int32)
    a, b = tf.while_loop(cond, body, [ii, v0])
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    y = sess.run(b)
    print(y)



def TFTest():
    sess = tf.Session()

    v0 = tf.Variable([[1, 2], [3, 4]], dtype=tf.float32)
    v1 = tf.Variable([[7, 8], [9, 3]], dtype=tf.float32)
    v2 = tf.Variable([4, 3, 4, 5, 6], dtype=tf.float32)
    v3 = tf.Variable([4, 9, 0, 3, 6], dtype=tf.float32)

    print(v0.shape)


    # v0 = tf.reshape(v0, [2, 1])
    # v1 = tf.reshape(v1, [1, 2])
    # v = tf.matmul(v0, v1)

    v = tf.concat([v0, v1], 1)    
    # v = v[:,1:3]

    a = v2[1:3]
    b = v2[2:4]
    c = v3[0:2]
    print(c.shape)

    v = tf.assign(c, a*b)
    print(v.shape)

    print(v3)
    print(v)
    v = v[:]
    print(v.shape)

    # v = tf.reverse(v0, [0])
    v = tf.concat([v0, v1], axis=1)
	
    # s = v.shape
    # print(s[0])

    sess.run(tf.global_variables_initializer())
    x = sess.run(v)
    print(x)
    
    # vs.variable



if __name__ == '__main__':
	TFTest()
    # LoopTest()
    # TensorArrayTest()


    # if False:
    #     a = TestClass()
    #     c = TestClass()
    #     c.z = 12

    #     c.foo()

    #     print(TestClass.__dict__)    
    #     print(TestClass.__name__)


    #     print(type(c.foo))

    #     # print(c.__str__)
    #     # print(c.__class__)

    #     # print(dir(c))