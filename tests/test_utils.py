import numpy as np
from objective_functions.recon import *
from tests.common import *

def test_AUPRC(set_seeds):
    from utils.AUPRC import ptsort, AUPRC
    assert ptsort([0,1,2]) == 0
    assert AUPRC([(0,0),(1,1)]) == 1.0

def test_weighted_acc(set_seeds):
    from utils.evaluation_metric import weighted_accuracy
    assert weighted_accuracy(np.array([1,1,1,1,1,0,0,0,0,0]), np.array([1,1,1,1,1,0,0,0,0,0])) == 1

def test_recon(set_seeds):
    assert sigmloss1d(torch.ones((10,10)),torch.ones((10,10))).shape == (10,)
    assert sigmloss1dcentercrop(10,10)(torch.ones((1,1,10,10)),torch.ones((1,1,10,10))).shape == (1,)
    assert nosigmloss1d(torch.ones((10,10)),torch.ones((10,10))).shape == (10,)