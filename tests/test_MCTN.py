from fusions.MCTN import *
from tests.common import *

def test_encoder(set_seeds):
    """Test Encoder module."""
    encoder = Encoder(10,2,1)
    assert encoder(torch.randn(10,10,10))[0].shape == (10,10,2)
    assert torch.isclose(norm:=torch.norm(encoder(torch.randn(10,10,10))[0]),torch.Tensor([8.868117332458496])), norm.item()
    assert count_parameters(encoder) == 168, count_parameters(encoder)

def test_attention(set_seeds):
    """Test Attention module."""
    attention = Attention(10,)
    assert attention(torch.randn(1,5,10),torch.randn(2,5,10))[0].shape == (1,2)
    assert torch.isclose(norm:=torch.norm(attention(torch.randn(1,5,10),torch.randn(2,5,10))),torch.Tensor([0.01771610416471958])), norm.item()
    assert count_parameters(attention) == 220, count_parameters(attention)

def test_decoder(set_seeds):
    """Test Decoder module."""
    decoder = Decoder(10,2,1)
    #assert decoder(torch.randn(1,10,2), torch.randn(1,10,2), torch.randn(1,10,2))[0].shape == (10,10,10)
    #assert torch.isclose(norm:=torch.norm(decoder(torch.randn(1,10,2), torch.randn(1,10), torch.randn(10,) )[0]),torch.Tensor([8.868117332458496])), norm.item()
    assert count_parameters(decoder) == 982, count_parameters(decoder)

def test_Seq2Seq(set_seeds):
    """Test Seq2Seq module."""
    encoder = Encoder(10,2,1)
    decoder = Decoder(10,2,1)
    seq2seq = Seq2Seq(encoder,decoder)
    #assert seq2seq(torch.randn(10,10,10), torch.randn(10,10,10))[0].shape == (10,10,10)
    #assert torch.isclose(norm:=torch.norm(seq2seq(torch.randn(10,10,10),torch.randn(10,10,10))[0]),torch.Tensor([8.868117332458496])), norm.item()
    assert count_parameters(seq2seq) == 1150, count_parameters(seq2seq)

