import torch
import torch.nn.functional as F
from torch import Tensor, nn

class SmoothNetResBlock(nn.Module):

    def __init__(self, in_channels, hidden_channels, dropout=0.5):
        super().__init__()
        self.linear1 = nn.Linear(in_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, in_channels)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(p=dropout, inplace=True)

    def forward(self, x):
        identity = x
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.lrelu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = self.lrelu(x)

        out = x + identity
        return out


class SmoothNet(nn.Module):

    def __init__(self,
                 window_size: int,
                 output_size: int,
                 hidden_size: int = 512,
                 res_hidden_size: int = 256,
                 num_blocks: int = 3,
                 dropout: float = 0.5):
        super().__init__()
        self.window_size = window_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.res_hidden_size = res_hidden_size
        self.num_blocks = num_blocks
        self.dropout = dropout

        assert output_size <= window_size, (
            'The output size should be less than or equal to the window size.',
            f' Got output_size=={output_size} and window_size=={window_size}')

        # Build encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(window_size, hidden_size),
            nn.LeakyReLU(0.1, inplace=True))

        # Build residual blocks
        res_blocks = []
        for _ in range(num_blocks):
            res_blocks.append(
                SmoothNetResBlock(
                    in_channels=hidden_size,
                    hidden_channels=res_hidden_size,
                    dropout=dropout))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Build decoder layers
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        N, C, T = x.shape
        x = x.to(torch.float32)

        assert T == self.window_size, (
            'Input sequence length must be equal to the window size. ',
            f'Got x.shape[2]=={T} and window_size=={self.window_size}')

        # Forward layers
        x = self.encoder(x)
        x = self.res_blocks(x)
        x = self.decoder(x)  # [N, C, output_size]
        return x


class MotionSmoothNet(nn.Module):

    def __init__(self, window_size: int, output_size: int):
        super().__init__()
        self.window_size = window_size
        self.output_size = output_size
        self.pos_smooth = SmoothNet(self.window_size, self.window_size - 2)
        self.v_smooth = SmoothNet(self.window_size - 1, self.window_size - 2)
        self.a_smooth = SmoothNet(self.window_size - 2, self.window_size - 2)
        self.fusion_layer = nn.Sequential(
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(3 * (self.window_size - 2), self.output_size)).to('cuda')

    def forward(self, x: Tensor) -> Tensor:
        pos_len = x.shape[2]
        pos_t1 = x[:, :, :(pos_len - 1)]
        pos_t2 = x[:, :, 1:pos_len]
        v = pos_t2 - pos_t1
        v_len = v.shape[2]
        v_t1 = v[:, :, :(v_len - 1)]
        v_t2 = v[:, :, 1:v_len]
        a = v_t2 - v_t1

        pos_fea = self.pos_smooth(x)
        v_fea = self.v_smooth(v)
        a_fea = self.a_smooth(a)
        fea_fusion = torch.concat((pos_fea, v_fea, a_fea), dim=-1)
        x = self.fusion_layer(fea_fusion)

        return x