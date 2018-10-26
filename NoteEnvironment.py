import os
import tensorflow as tf
from tensorflow.python.client import device_lib

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

def IsGPUAvailable():
    deviceList = device_lib.list_local_devices()
    if deviceList is None:
        return False

    for device in deviceList:
        if device.device_type == 'GPU':
            return True

    return False

EnvironmentVariableHasSet = False
def SetPrefrenceEnvironmentVariable():
    if EnvironmentVariableHasSet:
        print('[NoteEnvironment] environment variable has set.')
        return

    if IsLinux():
        # SetEnv('KMP_BLOCKTIME', 0)
        # SetEnv('KMP_SETTINGS', 'true')
        SetEnv('KMP_AFFINITY', 'granularity=fine,compact,1,0')
        SetEnv('OMP_NUM_THREADS', 4)
        SetEnv('MKL_NUM_THREADS', 1)
        # SetEnv('MKL_DYNAMIC', 'FALSE')
    else:
        print('[NoteEnvironment] windows system not set mkl env now.')
    globals()['EnvironmentVariableHasSet'] = True

def GenerateDefaultSessionConfig():
    config = None
    gpuOptions = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    if IsLinux():
        config = tf.ConfigProto(gpu_options=gpuOptions)
        config.intra_op_parallelism_threads = 4
        config.inter_op_parallelism_threads = 2
        print('[NoteEnvironment] tensorflow config', 
                'intra', config.intra_op_parallelism_threads, 
                'inter', config.inter_op_parallelism_threads, 
                'gpu per_process_gpu_memory_fraction', gpuOptions.per_process_gpu_memory_fraction)
    else:
        config = tf.ConfigProto(gpu_options=gpuOptions)
        print('[NoteEnvironment] tensorflow config', 
            'gpu per_process_gpu_memory_fraction', gpuOptions.per_process_gpu_memory_fraction)
    return config