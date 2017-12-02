import madmom
from madmom.ml.nn import NeuralNetworkEnsemble
from madmom.models import DOWNBEATS_BLSTM
import pickle

class TestClass():

    z = 111

    def foo(self):
        self.x = 10
        self.y = 20

    def __init__(self):
        self.bbb = 'asd'


def Do():    
    nn = NeuralNetworkEnsemble.load(DOWNBEATS_BLSTM)

    assert len(nn.processors) == 2
    assert type(nn.processors[0]) == madmom.processors.ParallelProcessor
    assert nn.processors[1] == madmom.ml.nn.average_predictions

    
    # pickle.dump(nn, open('d:/output/test.txt', 'wb'), protocol=0)
 
    print(nn)
    # for i in dir(nn):
    #     print(i)

    for i in nn.processors:
        print(i)


if __name__ == '__main__':

    if True:
        Do()

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