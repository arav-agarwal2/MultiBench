from objective_functions.contrast import *
import numpy as np
from tests.common import *

def test_Alias(set_seeds):
    alias = AliasMethod(torch.tensor([3.0,1.0,2.0]).float())
    assert alias.draw(10).shape == (10,)
    assert torch.norm(alias.draw(10).float()).item() == 4.582575798034668

def test_MutlSimLoss(set_seeds):
    loss = MultiSimilarityLoss()
    assert loss(torch.zeros((10,10)),torch.cat([torch.zeros((5,10)),torch.ones((5,10))],dim=0)).shape == ()
    assert loss(torch.zeros((10,10)),torch.zeros((10,10))).item() == 0

def test_NCESoftmaxLoss(set_seeds):
    loss = NCESoftmaxLoss()
    assert loss(torch.ones((10,10))).shape == ()
    assert np.isclose(loss(torch.ones((10,10))).item(),2.3026)
    loss = NCECriterion(10)
    assert loss(torch.ones((10,10))).shape == (1,)
    assert np.isclose(loss(torch.ones((10,10))).item(),7.3668)
    NCEAverage(3,4,1)