
import unimodals.common_models as cm
import unimodals.robotics.encoders as robenc
import unimodals.robotics.decoders as robdec
import torch
from tests.common import *

def test_LSTM(set_seeds):
    """Test LSTM shape."""
    test = torch.randn([32, 50, 35])
    model = cm.LSTM(35,2, linear_layer_outdim=2, dropout=True)
    assert model(test).shape == (32,2)
    assert torch.isclose(norm := torch.norm(model(test)),torch.Tensor([3.777543306350708])), norm.item()



def test_VGG(set_seeds):
    """Test VGG shape."""
    test = torch.randn([32, 3, 128, 128])
    model = cm.VGG(35)
    assert model(test)[0].shape == (32,512)
    assert torch.isclose(norm := torch.norm(model(test)[0]),torch.Tensor([282.1166076660156])), norm.item()



def test_ResNetLSTM(set_seeds):
    """Test ResNetLSTM shape."""
    test = torch.randn([1,3, 150, 112, 112])
    model = cm.ResNetLSTMEnc(10, dropout=True)
    assert model(test).shape == (1,10)
    assert torch.isclose(norm := torch.norm(model(test)),torch.Tensor([1.6341464519500732])), norm.item()


def test_robotics_encoders(set_seeds):
    enc = robenc.ActionEncoder(10)
    assert enc(torch.randn(32,10)).shape == (32,32)
    assert torch.isclose(norm := torch.norm(enc(torch.randn(32,10))),torch.Tensor([7.25303316116333])), norm.item()
    assert enc(None) is None
    enc = robenc.DepthEncoder(10,0.2)
    assert enc(torch.randn(32,1,128,128))[0].shape == (32,20,1)
    assert torch.isclose(norm := torch.norm(enc(torch.randn(32,1,128,128))[0]),torch.Tensor([1.751620888710022])), norm.item()
    enc = robenc.ImageEncoder(10,0.2)
    assert enc(torch.randn(32,128,128,3))[0].shape == (32,20,1)
    assert torch.isclose(norm := torch.norm(enc(torch.randn(32,128,128,3))[0]),torch.Tensor([2.152540922164917])), norm.item()
    enc = robenc.ForceEncoder(10,0.2)
    assert enc(torch.randn(32,6,32))[0].shape == (20,1)
    assert torch.isclose(norm := torch.norm(enc(torch.randn(32,6,32))[0]),torch.Tensor([0.08203933387994766])), norm.item()
    enc = robenc.ProprioEncoder(10,0.2)
    assert enc(torch.randn(32,8))[0].shape == (20,1)
    assert torch.isclose(norm := torch.norm(enc(torch.randn(32,8))[0]),torch.Tensor([0.15822094678878784])), norm.item()
    
def test_res3d2(set_seeds):
    from functools import reduce
    def get_param_count(model):
        return sum(reduce(lambda a,b: a*b, x.size()) for x in model.parameters() if x.requires_grad)
    from unimodals.res3d import generate_model, ResNet, BasicBlock, _get_inplanes
    assert get_param_count(generate_model(18)) == 33421164
    assert get_param_count(generate_model(34)) == 63747500
    model = generate_model(50)
    assert model(torch.randn((1,3,3,128,128))).shape == (1,400)
    assert torch.isclose(norm := torch.norm(model(torch.randn(1,3,3,128,128))),torch.Tensor([11.84746265411377])), norm.item()
    assert get_param_count(generate_model(50)) == 47035382
    assert get_param_count(generate_model(101)) == 86101110
    assert get_param_count(generate_model(152)) == 118277110
    assert get_param_count(generate_model(200)) == 127485942
    assert get_param_count(ResNet(BasicBlock, [1, 1, 1, 1], _get_inplanes(),shortcut_type='A')) == 14434220
    model = ResNet(BasicBlock, [1, 1, 1, 1], _get_inplanes(),shortcut_type='A')
    assert model(torch.randn((1,3,3,128,128))).shape == (1,400)
    assert torch.isclose(norm := torch.norm(model(torch.randn(1,3,3,128,128))),torch.Tensor([10.420717239379883])), norm.item()
    
    