import os
import tensorflow as tf

def GetEnv(key):
    if key not in os.environ:
        return '(not_defined)'
    return os.environ[key]

def SetEnv(key, val):
    old = GetEnv(key)
    val = str(val)
    os.environ[key] = val
    print('[NoteEnvironment] python set env ', key, old, '=>', val)

def IsLinux():
    return 'nt' != os.name

def SetPrefrenceEnvironmentVariable():
    if IsLinux():
        # SetEnv('KMP_BLOCKTIME', 0)
        # SetEnv('KMP_SETTINGS', 'true')
        SetEnv('KMP_AFFINITY', 'granularity=fine,compact,1,0')
        SetEnv('OMP_NUM_THREADS', 4)
        SetEnv('MKL_NUM_THREADS', 1)
        # SetEnv('MKL_DYNAMIC', 'FALSE')
    else:
        print('[NoteEnvironment] windows system not set mkl env now.')

def GenerateDefaultSessionConfig():
    config = None
    if IsLinux():
        config = tf.ConfigProto()
        config.intra_op_parallelism_threads = 4
        config.inter_op_parallelism_threads = 2
        print('[NoteEnvironment] tensorflow config', 'intra', config.intra_op_parallelism_threads, 'inter', config.inter_op_parallelism_threads)
    else:
        print('[NoteEnvironment] windows system not set tensorflow config now.')
    return config