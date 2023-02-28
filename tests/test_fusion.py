from fusions.common_fusions import *
from tests.common import *


def test_concat(set_seeds):
    """Test concat."""
    fusion = Concat()
    output = fusion([torch.randn((1,2)) for _ in range(2)])
    assert output.shape == (1,4)
    assert torch.isclose(torch.norm(output),torch.Tensor([2.7442679405212402])), torch.norm(output).item()
    assert count_parameters(fusion) == 0, count_parameters(fusion)

def test_concat_early(set_seeds):
    """Test concat early."""
    fusion = ConcatEarly()
    output = fusion([torch.randn((1,2,2)) for _ in range(2)])
    assert output.shape == (1,2,4) 
    assert torch.isclose(torch.norm(output),torch.Tensor([3.395326614379883])), torch.norm(output).item()
    assert count_parameters(fusion) == 0, count_parameters(fusion)

def test_stack(set_seeds):
    """Test stack."""
    fusion = Stack()
    output = fusion([torch.randn((1,2,2)) for _ in range(2)])
    assert output.shape == (1,4,2)
    assert torch.isclose(torch.norm(output),torch.Tensor([3.395326614379883])), torch.norm(output).item()
    assert count_parameters(fusion) == 0, count_parameters(fusion)
    
def test_concat_linear(set_seeds):
    """Test concat linear."""
    fusion = ConcatWithLinear(2,2)
    output = fusion([torch.randn((1,2,2)) for _ in range(2)])
    assert output.shape == (1,4,2)
    assert torch.isclose(torch.norm(output),torch.Tensor([0.9164801836013794])), torch.norm(output).item()
    assert count_parameters(fusion) == 6, count_parameters(fusion)
    
def test_tensor_fusion(set_seeds):
    """Test tensor fusion."""
    fusion = TensorFusion()
    output = fusion([torch.randn((1,2,2)) for _ in range(2)])
    assert output.shape == (1,2,9)
    assert torch.isclose(torch.norm(output),torch.Tensor([5.0617828369140625])), torch.norm(output).item()
    assert count_parameters(fusion) == 0, count_parameters(fusion)

def test_low_rank_tensor_fusion(set_seeds):
    """Test low rank tensor fusion."""
    fusion = LowRankTensorFusion((10,10),2,1)
    output = fusion([torch.randn((10,10)) for _ in range(2)])
    assert output.shape == (10,2)
    assert torch.isclose(torch.norm(output),torch.Tensor([0.6151207089424133])), torch.norm(output).item()
    assert count_parameters(fusion) == 3, count_parameters(fusion)
    
def test_multiplicative_interaction_models(set_seeds):
    """Test multiplicative interaction models."""
    from unimodals.common_models import LeNet, MLP, Constant
    import torch
    from fusions.common_fusions import MultiplicativeInteractions2Modal
    data = [torch.randn((64,1,28,28)),torch.randn((64,1,112,112)),torch.cat((torch.ones((32,)),torch.randn((32,))),dim=0).long()]

    channels = 1
    encoders = [LeNet(1, channels, 3), LeNet(1, channels, 5)]
    head = MLP(channels*40, 100, 10)

    # fusion=Concat().cuda()
    fusion = MultiplicativeInteractions2Modal(
        [channels*8, channels*32], channels*40, 'matrix')


    out = fusion([encoders[0](data[0]), encoders[1](data[1])])
    assert out.shape == (64,40)
    assert torch.isclose(torch.norm(out),torch.Tensor([105.80758666992188])), torch.norm(out).item()
    assert count_parameters(fusion) == 11880, count_parameters(fusion)


    fusion = MultiplicativeInteractions2Modal(
        [channels*8, channels*32], [channels,channels], 'matrix3D')

    out = fusion([encoders[0](data[0]), encoders[1](data[1])])
    assert out.shape == (64,1,1)
    assert torch.isclose(torch.norm(out),torch.Tensor([95.81939697265625])), torch.norm(out).item()
    assert count_parameters(fusion) == 297, count_parameters(fusion)

    fusion = MultiplicativeInteractions2Modal(
        [channels*8, channels*32], channels*40, 'vector')

    out = fusion([encoders[0](data[0]), encoders[1](data[1])])
    assert out.shape == (64,32)
    assert torch.isclose(torch.norm(out),torch.Tensor([77.0551986694336])), torch.norm(out).item()
    assert count_parameters(fusion) == 576, count_parameters(fusion)

    fusion = MultiplicativeInteractions2Modal(
        [channels*8, channels*32], channels*40, 'scalar', grad_clip=(0.1,0.2))

    out = fusion([encoders[0](data[0]), encoders[1](data[1])])
    assert out.shape == (64,32)
    assert torch.isclose(torch.norm(out),torch.Tensor([156.4239501953125])), torch.norm(out).item()
    assert count_parameters(fusion) == 18, count_parameters(fusion)


    fusion = MultiplicativeInteractions3Modal(
        [channels*8, channels*32, channels*8], channels*40)

    out = fusion([encoders[0](data[0]), encoders[1](data[1]),encoders[0](data[0])])
    assert out.shape == (64,64,40)
    assert torch.isclose(torch.norm(out),torch.Tensor([1006.2703857421875])), torch.norm(out).item()
    assert count_parameters(fusion) == 106920, count_parameters(fusion)


def test_nl_gate(set_seeds):
  from fusions.common_fusions import NLgate
  fusion = NLgate(24, 30, 10, None, (10, 300), (10, 300))
  out = fusion([torch.randn((24,30)),torch.randn((30,10))])
  assert out.shape == (30,720)
  assert torch.isclose(torch.norm(out),torch.Tensor([162.23263549804688])), torch.norm(out).item()
  assert count_parameters(fusion) == 6600, count_parameters(fusion)

def test_MULTModel(set_seeds):
    from fusions.mult import MULTModel

    class HParams():
            num_heads = 8
            layers = 4
            attn_dropout = 0.1
            attn_dropout_modalities = [0,0,0.1]
            relu_dropout = 0.1
            res_dropout = 0.1
            out_dropout = 0.1
            embed_dropout = 0.2
            embed_dim = 40
            attn_mask = True
            output_dim = 1
            all_steps = False

    data = [torch.randn([32, 50, 20]), torch.randn([32, 50, 5]), torch.randn([32, 50, 300]), torch.cat((torch.ones((16,1)),torch.randn((16,1))),dim=0).long()]

    fusion = MULTModel(3, [20, 5, 300], hyp_params=HParams)
    out = fusion([data[0],data[1],data[2]])
    assert out.shape == (32,1)
    assert torch.isclose(torch.norm(out),torch.Tensor([4.186001300811768])), torch.norm(out).item()
    assert count_parameters(fusion) == 3076961, count_parameters(fusion)

