import tensorflow as tf
import sys
from tensorflow.python.client import device_lib

def postProcessString(val):
    return val.replace('\\', '/')

def outputResult(val):
    print(val, end='')

def getIncludeDir():
    return postProcessString(tf.sysconfig.get_include())

def getLibDir():
    return postProcessString(tf.sysconfig.get_lib())

def getCompileFlags():
    return postProcessString(" ".join(tf.sysconfig.get_compile_flags()))

def getLinkFlags():
    return postProcessString(" ".join(tf.sysconfig.get_link_flags()))

def checkDevice():
    deviceList = device_lib.list_local_devices()
    res ='CPU'
    if deviceList is not None:
        for device in deviceList:
            if device.device_type == 'GPU':
                res = 'GPU'
                break
    return res

if __name__ == '__main__':
    func = globals()[sys.argv[1]]
    outputResult(func())
