import tensorflow as tf
import sys

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

if __name__ == '__main__':
    func = globals()[sys.argv[1]]
    outputResult(func())
