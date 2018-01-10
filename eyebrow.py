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
batch_size = 4
n_classes = 3


def LoadPicture(pathname, reverse, session):
    
    image_raw_data_png = tf.gfile.FastGFile(pathname, 'rb').read()
    img = tf.image.decode_png(image_raw_data_png, 1)
    if reverse:
        img = tf.reverse(img, axis=[1])
        
    img = tf.cast(img, tf.float32)
    img = tf.div(img, 255.0)

    r = sess.run(img)
    return r


# @run
def PrepareData():

    sess = tf.Session()        
    files = os.listdir(Path)

    pics = []
    lables = []

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

        revert = (side == SIDE_LEFT)
        print(f, 'revert', revert)
        


@run
def BuildNetwork():
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet'):
        # TF Estimator input is a dict, in case of multiple inputs
        
        x = tf.placeholder(dtype=tf.float32, shape=[batch_size, 30, 90, 1])

        # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        # x = tf.reshape(x, shape=[-1, 28, 28, 1])

        # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
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
        # fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)
        print(out)
        
        pred_classes = tf.argmax(out, axis=1)
        pred_probas = tf.nn.softmax(out)

    return out


def DoWork():
    pass



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
