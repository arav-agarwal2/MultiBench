from fusions.MVAE import *
from unimodals.MVAE import *
from tests.common import *

def test_product_of_experts(set_seeds):
    """Test ProductOfExperts module."""
    poe = ProductOfExperts((2,2,2))
    assert poe(torch.randn(10,2,2),torch.zeros(10,2,2))[0].shape == (2,2)
    assert torch.isclose(norm:=torch.norm(poe(torch.randn(10,2,2),torch.zeros(10,2,2))[0]),torch.Tensor([0.3183247745037079])), norm.item()
    assert count_parameters(poe) == 0, count_parameters(poe)

    poe2 = ProductOfExperts_Zipped((2,2,2))
    assert poe2(torch.randn(10,2,2,2))[0].shape == (2,2)
    assert torch.isclose(norm:=torch.norm(poe2(torch.randn(10,2,2,2))[0]),torch.Tensor([0.4636370837688446])), norm.item()
    assert count_parameters(poe2) == 0, count_parameters(poe2)


def test_mlp_encoder(set_seeds):
    """Test MLPEncoder module."""
    mlp = MLPEncoder(10,2,1)
    assert mlp(torch.randn(10,10))[0].shape == (10,1)
    assert torch.isclose(norm:=torch.norm(mlp(torch.randn(10,10))[0]),torch.Tensor([2.285719394683838])), norm.item()
    assert count_parameters(mlp) == 28, count_parameters(mlp)
