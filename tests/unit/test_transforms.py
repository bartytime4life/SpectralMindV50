import torch
from spectramind.data.transforms import zscore_time, minmax_time, detrend_time_linear, clip, Compose

def _toy_bt(device=None, dtype=torch.float32):
    x = torch.tensor([[1., 2., 3., 4.],
                      [10., 10., 10., 10.]], device=device, dtype=dtype)
    m = torch.tensor([[1, 1, 1, 1],
                      [0, 0, 0, 0]], device=device, dtype=torch.float32)
    return x, m

def _toy_btc(device=None, dtype=torch.float32):
    x = torch.stack([
        torch.stack([torch.arange(4., device=device), torch.arange(4., device=device)] , dim=-1),  # [T=4,C=2]
        torch.zeros( (4,2), device=device )
    ], dim=0)  # [B=2,T=4,C=2]
    m = torch.tensor([[1, 1, 1, 1],
                      [0, 0, 0, 0]], device=device, dtype=torch.float32)
    return x.to(dtype), m

def test_zscore_bt_all_masked_row_safe():
    x, m = _toy_bt()
    z = zscore_time(x, m)
    assert torch.allclose(z[0].mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.all(z[1] == 0)

def test_minmax_bt_all_masked_row_safe():
    x, m = _toy_bt()
    y = minmax_time(x, m, 0.0, 1.0)
    assert y[0].min() >= 0 and y[0].max() <= 1
    assert torch.all(y[1] == 0.0)  # becomes min_val

def test_detrend_bt_all_masked_row_safe():
    x, m = _toy_bt()
    r = detrend_time_linear(x, m)
    assert abs(r[0].sum().item()) < 1e-4
    assert torch.all(r[1] == 0.0)

def test_zscore_btc_channelwise_and_mask():
    x, m = _toy_btc()
    z = zscore_time(x, m)
    # first batch has two channels identical 0..3 -> same z-stats
    assert torch.allclose(z[0,:,0], z[0,:,1], atol=1e-6)
    # second batch fully masked -> zeros
    assert torch.all(z[1] == 0)

def test_compose_pipeline():
    x, m = _toy_bt()
    pipe = Compose([
        lambda x, mask=None: detrend_time_linear(x, mask=mask),
        lambda x, mask=None: zscore_time(x, mask=mask),
        lambda x, **k: clip(x, -3, 3),
    ])
    y = pipe(x, mask=m)
    assert y.shape == x.shape
