# models/tsm.py
import torch
import torch.nn as nn

def temporal_shift(x, num_frames, shift_ratio=0.125):
    """
    x: (B*T, C, H, W)
    """
    BT, C, H, W = x.shape
    assert BT % num_frames == 0, "Batch must be B*T"

    B = BT // num_frames
    x = x.view(B, num_frames, C, H, W)

    fold = int(C * shift_ratio)
    out = torch.zeros_like(x)

    # shift backward (t-1)
    out[:, 1:, :fold] = x[:, :-1, :fold]

    # shift forward (t+1)
    out[:, :-1, fold:2*fold] = x[:, 1:, fold:2*fold]

    # static channels
    out[:, :, 2*fold:] = x[:, :, 2*fold:]

    return out.view(BT, C, H, W)

class ZeroConvTSM(nn.Module):
    def __init__(self, channels, num_frames):
        super().__init__()
        self.num_frames = num_frames

        self.zero_conv = nn.Conv2d(
            channels, channels, kernel_size=1
        )

        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)

    def forward(self, x):
        # 🔥 Handle tuple outputs from attention
        if isinstance(x, tuple):
            hidden_states = x[0]
        else:
            hidden_states = x

        tsm_out = temporal_shift(hidden_states, self.num_frames)
        hidden_states = hidden_states + self.zero_conv(tsm_out)

        # 🔁 Repack tuple if needed
        if isinstance(x, tuple):
            return (hidden_states,) + x[1:]
        else:
            return hidden_states
