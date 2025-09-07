from spectramind.utils.seed import set_seed
import torch
def test_seed():
    set_seed(1337)
    a = torch.randn(3)
    set_seed(1337)
    b = torch.randn(3)
    assert torch.allclose(a, b)
