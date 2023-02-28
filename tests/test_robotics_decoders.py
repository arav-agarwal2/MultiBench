from unimodals.robotics.decoders import *
from tests.common import *


def test_optical(set_seeds):
    ofd = OpticalFlowDecoder(4)

def test_eedelta(set_seeds):
    eed = EeDeltaDecoder(4,10)
    assert eed(torch.zeros(10,4)).shape == (10,10)

def test_contact(set_seeds):
    cd = ContactDecoder(10,True)
    assert cd((None, torch.Tensor(10,10),None, None)).shape == (10,1)
    cd = ContactDecoder(10,False)
    assert cd((None, torch.Tensor(10,10),None, None,None,None,None,None)).shape == (10,1)