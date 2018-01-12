import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network.py#L113

runList = []
def run(r):
    runList.append(r)
    return r

Path = r'D:/work/eyebrow/'
SIDE_LEFT = 0
SIDE_RIGHT = 1
# Training Parameters
learning_rate = 0.001
num_steps = 2000
batch_size = 8
n_classes = 3


def LoadPicture(pathname, reverse, session):
    
    image_raw_data_png = tf.gfile.FastGFile(pathname, 'rb').read()
    img = tf.image.decode_png(image_raw_data_png, 1)
    if reverse:
        img = tf.reverse(img, axis=[1])
        
    img = tf.cast(img, tf.float32)
    img = tf.div(img, 255.0)

    r = session.run(img)
    return r


# @run
def PrepareData():

    sess = tf.Session()        
    files = os.listdir(Path)

    pics = []
    lables = []
    
    print('start loading')
    for f in files:
        # print(f)
        pathname = Path + f
        r = os.path.splitext(f)[0].split('_')
        eClass = int(r[-1])
        side = r[-3]
        assert side == 'leb' or side == 'reb'
        if side == 'leb':
            side = SIDE_LEFT
        if side == 'reb':
            side = SIDE_RIGHT

        reverse = (side == SIDE_LEFT)
        p = LoadPicture(pathname, reverse, sess)
        pics.append(p)

        l = np.zeros(n_classes)
        l[eClass-1] = 1.0
        lables.append(l)
    
    pics = np.array(pics)
    lables = np.array(lables)
    print(len(files), 'pics loaded')
    return pics, lables        


# @run
def BuildNetwork(X, labels):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet'):
        # TF Estimator input is a dict, in case of multiple inputs
        
        # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        # x = tf.reshape(x, shape=[-1, 28, 28, 1])

        # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(X, 32, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        # Convolution Layer with 64 filters and a kernel size of 3
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)

        # Fully connected layer (in tf contrib folder for now)
        fc1 = tf.layers.dense(fc1, 1024)
        # Apply Dropout (if is_training is False, dropout is not applied)
        # fc2 = tf.layers.dropout(fc1, rate=0.25, training=True)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)

        # out_test = tf.layers.dense(fc1, n_classes)
        
        # pred_classes = tf.argmax(out, axis=1)
        # print(pred_classes)
        pred_probas = tf.nn.softmax(out)
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=labels))
 
        print('create optimizer')
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op)
        
        pred_probas = tf.nn.softmax(out)
        correct = tf.equal(tf.argmax(pred_probas, axis=1), tf.argmax(labels, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        # acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)
        # print('network created')

    return train_op, loss_op, accuracy

@run
def DoWork():
    
    X = tf.placeholder(dtype=tf.float32, shape=[batch_size, 30, 90, 1])
    Y = tf.placeholder(dtype=tf.float32, shape=[batch_size, 3])

    train_op, loss_op, accuracy_op = BuildNetwork(X, Y)

    data, lables = PrepareData()
    rou = len(data) % batch_size
    batch_count = len(data) // batch_size
    if rou != 0:
        data = data[0:-rou]
        lables = lables[0:-rou]
    
    data = np.reshape(data, [-1, batch_size, 30, 90, 1])
    lables = np.reshape(lables, [-1, batch_size, 3])

    # train_data = data[0:-3]
    # train_lables = lables[0:-3]

    # test_data = data[-3:]
    # test_lables = lables[-3:]

    train_data = data[3:]
    train_lables = lables[3:]

    test_data = data[:3]
    test_lables = lables[:3]

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print('start training')
    epch = 200
    for e in range(epch):
        loss = []
        acc = []

        if e % 10 == 0:
            for i in range(len(test_data)):
                l, a = sess.run([loss_op, accuracy_op], feed_dict={X:test_data[i], Y:test_lables[i]})
                loss.append(l)
                acc.append(a)
            print('epch', e, 'loss', sum(loss) / len(loss), 'accuracy', sum(acc) / len(acc))

            # shuffle data
            train_data = np.reshape(train_data, [-1, 30, 90, 1])
            train_lables = np.reshape(train_lables, [-1, 3])

            shuffle = np.random.permutation(len(train_data))
            train_data = train_data[shuffle]
            train_lables = train_lables[shuffle]
            train_data = np.reshape(train_data, [-1, batch_size, 30, 90, 1])
            train_lables = np.reshape(train_lables, [-1, batch_size, 3])
            continue

        for i in range(len(train_data)):
            t, l = sess.run([train_op, loss_op], feed_dict={X:train_data[i], Y:train_lables[i]})
            loss.append(l)

        print('epch', e, 'loss', sum(loss) / len(loss))


# @run
def ListFile():

    files = os.listdir(Path)
    # for f in files:
        # print(f)

    image_raw_data_png = tf.gfile.FastGFile(Path + files[0], 'rb').read()
    img = tf.image.decode_png(image_raw_data_png, 1)  #Tensor
    img = tf.reverse(img, axis=[1])
    print(img)
    print(files[0])

    sess = tf.Session()
    img = tf.cast(img, tf.float32)
    y = tf.div(img, 255.0)

    y = sess.run(y)
    print(type(y))
    print(y.shape)
    print(y)
    
    # y = np.repeat(y, 3, axis=2)

    # plt.figure(1)
    # plt.imshow(y)
    # plt.show()

    # print()
    # print(type(image_raw_data_png), len(image_raw_data_png))


if __name__ == '__main__':
    
    for fun in runList:
        fun()
