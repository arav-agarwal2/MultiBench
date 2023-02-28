from unimodals.common_models import *
import torch
from tests.common import *

DATA_PATH = '/home/arav/MultiBench/MultiBench/'

def test_id(set_seeds):
    """Test Identity module."""
    id = Identity()
    test = torch.Tensor([0])
    assert id(test) == test
    
def test_linear(set_seeds):
    """Test Linear Module."""
    lin = Linear(3,4)
    test = torch.randn(4,3)
    assert lin(test).shape == (4,4)
    assert torch.isclose(torch.norm(lin(test)),torch.Tensor([1.876538872718811])), torch.norm(lin(test)).item()
    assert count_parameters(lin) == 16, count_parameters(lin)
    
    lin = Linear(3,4, True)
    test = torch.randn(4,3)
    assert lin(test).shape == (4,4)
    assert torch.isclose(torch.norm(lin(test)),torch.Tensor([2.4047462940216064])), torch.norm(lin(test)).item()
    assert count_parameters(lin) == 16, count_parameters(lin)
    
def test_squeeze(set_seeds):
    """Test squeeze module."""
    lin = Squeeze(1)
    test = torch.randn((4,1))
    assert lin(test).shape == (4,)
    assert torch.isclose(torch.norm(lin(test)),torch.Tensor([2.7442679405212402])), torch.norm(lin(test)).item()
    assert count_parameters(lin) == 0, count_parameters(lin)
    lin = Squeeze()
    assert lin(test).shape == (4,)
    assert torch.isclose(torch.norm(lin(test)),torch.Tensor([2.7442679405212402])), torch.norm(lin(test)).item()
    assert count_parameters(lin) == 0, count_parameters(lin)
    
def test_sequential(set_seeds):
    """Test sequential module."""
    lin = Sequential(Linear(1,2), Squeeze())
    test = torch.randn((1,))
    assert lin(test, training=True).shape == (2,)
    assert torch.isclose(torch.norm(lin(test, training=True)),torch.Tensor([2.068535566329956])), torch.norm(lin(test, training=True)).item()
    assert count_parameters(lin) == 4, count_parameters(lin)
    
def test_reshape(set_seeds):
    """Test Reshape module."""
    lin = Reshape((4,4))
    test = torch.randn((16,))
    assert lin(test).shape == (4,4)
    assert torch.isclose(torch.norm(lin(test)),torch.norm(test))
    assert count_parameters(lin) == 0, count_parameters(lin)


def test_transpose(set_seeds):
    """Test Transpose module."""
    lin = Transpose(0,1)
    test = torch.randn((3,4))
    assert lin(test).shape == (4,3)
    assert torch.isclose(torch.norm(lin(test)),torch.norm(test))
    assert count_parameters(lin) == 0, count_parameters(lin)
    
def test_mlp(set_seeds):
    """Test common module."""
    lin = MLP(3,2,1, True, 0.1,True)
    test = torch.randn((3,3))
    out = lin(test)
    assert out[0] == 0
    assert out[1].shape == test.shape
    assert out[2].shape == (3,2)
    assert out[3].shape == (3,2)
    sample_outs = [2.070115804672241,0.537086546421051, 0.5025997161865234]
    assert count_parameters(lin) == 11, count_parameters(lin)
    for compareto, elem in zip(sample_outs, out[1:]):
        assert torch.isclose(torch.norm(elem),torch.Tensor([compareto])), torch.norm(elem).item()
    lin = MLP(3,2,1)
    assert lin(test).shape == (3,1)
    assert torch.isclose(torch.norm(lin(test)),torch.Tensor([1.2006075382232666])), torch.norm(lin(test)).item()
    assert count_parameters(lin) == 11, count_parameters(lin)

    
def test_GRU(set_seeds):
    """Test common module."""
    lin = GRU(3,2,1, True)
    test = torch.randn((3,3,3))
    out = lin(test)
    assert out[0].shape == (3,2)
    assert torch.isclose(torch.norm(out),torch.Tensor([0.0])), torch.norm(out).item()
    assert count_parameters(lin) == 42, count_parameters(lin)
    lin.flatten = True
    assert lin(test).shape == (3,6)
    assert torch.isclose(torch.norm(lin(test)),torch.Tensor([0.0])), torch.norm(lin(test)).item()
    assert count_parameters(lin) == 42, count_parameters(lin)
    lin.last_only = True
    assert lin(test).shape == (3,2)
    assert torch.isclose(torch.norm(lin(test)),torch.Tensor([1.257783055305481])), torch.norm(lin(test)).item()
    assert count_parameters(lin) == 42, count_parameters(lin)

def test_Constant(set_seeds):
    """Test constant module."""
    cons = Constant(1)
    test = torch.randn((3,3))
    assert cons(test).shape == (1,)
    assert cons(test)[0] == 0
    assert torch.isclose(torch.norm(cons(test)),torch.Tensor([0.0])), torch.norm(cons(test)).item()
    assert count_parameters(cons) == 0, count_parameters(cons)

def test_DAN(set_seeds):
    """Test DAN."""
    test = torch.randn((2,4))
    model = DAN(4,2)
    assert model(test).shape == (2,)
    assert torch.isclose(torch.norm(model(test)),torch.Tensor([0.6967308521270752])), torch.norm(model(test)).item()
    assert count_parameters(model) == 28, count_parameters(model)


def test_Transformer(set_seeds):
    """Test Transformer Shape."""
    test = torch.randn((2,40,10))
    model = Transformer(10,10)
    assert model(test).shape == (2,10)
    assert torch.isclose(torch.norm(model(test)),torch.Tensor([4.472121715545654])), torch.norm(model(test)).item()
    assert count_parameters(model) == 217590, count_parameters(model)


def test_GlobalPooling(set_seeds):
    """Test Module."""
    test = torch.randn((2,40,40))
    model = GlobalPooling2D()
    assert model(test).shape == (2,40)
    assert torch.isclose(torch.norm(model(test)),torch.Tensor([1.374691128730774])), torch.norm(model(test)).item()
    assert count_parameters(model) == 0, count_parameters(model)

def test_MaxOut_MLP(set_seeds):
    """Test Module."""
    test = torch.randn((2,40))
    model = MaxOut_MLP(10, number_input_feats=40)
    assert model(test).shape == (2,10)
    assert torch.isclose(norm := torch.norm(model(test)),torch.Tensor([3.8609466552734375])), norm.item()
    assert count_parameters(model) == 14554, count_parameters(model)

def test_MaxOut(set_seeds):
    """Test Module."""
    test = torch.randn((2,10))
    model = Maxout(10,10,1)
    assert model(test).shape == (2,10)
    assert torch.isclose(torch.norm(model(test)),torch.Tensor([2.98047137260437])), torch.norm(model(test)).item()
    assert count_parameters(model) == 110, count_parameters(model)

def test_VGG(set_seeds):
    """Test Module."""
    test = torch.randn((1,3,128,128))
    model = VGG16(10)
    assert model(test).shape == (1,10)
    assert torch.isclose(norm := torch.norm(model(test)),torch.Tensor([1.0593265295028687])), norm.item()
    assert count_parameters(model) == 134309962, count_parameters(model)

    model = VGG16Slim(10)
    assert model(test).shape == (1,10)
    assert torch.isclose(norm := torch.norm(model(test)),torch.Tensor([0.8343296647071838])), norm.item()
    assert count_parameters(model) == 14974026, count_parameters(model)

    model = VGG11Slim(10)
    assert model(test).shape == (1,10)
    assert torch.isclose(norm := torch.norm(model(test)),torch.Tensor([0.7696740627288818])), norm.item()
    assert count_parameters(model) ==  250890, count_parameters(model)

    model = VGG11Pruned(10)
    assert model(test).shape == (1,10)
    assert torch.isclose(norm := torch.norm(model(test)),torch.Tensor([2.4355552196502686])), norm.item()
    assert count_parameters(model) == 641226, count_parameters(model)

    model = VGG16Pruned(10)
    assert model(test).shape == (1,10)
    assert torch.isclose(norm := torch.norm(model(test)),torch.Tensor([2.627531051635742])), norm.item()
    assert count_parameters(model) == 985626, count_parameters(model)
    
def test_LeNet(set_seeds):
    """Test module."""
    test = torch.randn((1,3,128,128))
    model = LeNet(3,2,1)
    assert model(test).shape == (4,32,32)
    assert torch.isclose(torch.norm(model(test)),torch.Tensor([77.07115173339844])), torch.norm(model(test)).item()
    assert count_parameters(model) == 234, count_parameters(model)
    

def test_Seq(set_seeds):
    """Test stuff."""
    test = torch.randn((8,3,3))
    lin = GRUWithLinear(3,2,1, True)
    assert lin(test).shape == (8,3,1)
    assert torch.isclose(norm := torch.norm(lin(test)),torch.Tensor([2.1098499298095703])), norm.item()
    assert count_parameters(lin) == 45, count_parameters(lin)
    lin = TwoLayersLSTM(3,3,True)
    assert lin(test).shape == (8,3,6)
    assert torch.isclose(norm := torch.norm(lin(test)),torch.Tensor([2.101766347885132])), norm.item()
    assert count_parameters(lin) == 468, count_parameters(lin)
    
def test_Resnet3d(set_seeds):
    """Test stuff."""
    from unimodals.res3d import generate_model
    test = torch.randn((3,3,128,128,128))
    lin = generate_model(10)
    assert lin(test).shape == (3,400)
    assert torch.isclose(torch.norm(lin(test)),torch.Tensor([11.619596481323242])), torch.norm(lin(test)).item()
    assert count_parameters(lin) ==  14608044, count_parameters(lin)

