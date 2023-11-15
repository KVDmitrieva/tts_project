import torch
from torch import nn

from hw_asr.base import BaseModel


class JasperSubmodule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, p=0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation),
            nn.BatchNorm1d(num_features=out_channels)
        )

        self.tail = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=p)
        )

    def forward(self, x, residual=None):
        x = self.head(x)

        if residual is not None:
            x += residual

        return self.tail(x)


class JasperBlock(nn.Module):
    def __init__(self, submodules_num, in_channels, out_channels,
                 kernel_size, stride=1, dilation=1, p=0.1):
        super().__init__()
        self.submodules = nn.ModuleList([
            JasperSubmodule(in_channels=in_channels if i == 0 else out_channels,
                            out_channels=out_channels, kernel_size=kernel_size,
                            stride=stride, padding=kernel_size * dilation // 2,
                            dilation=dilation, p=p) for i in range(submodules_num)
        ])
        self.residual = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels)
        )



    def forward(self, x):
        residual = self.residual(x)
        for submodule in self.submodules[:-1]:
            x = submodule(x)

        return self.submodules[-1](x, residual)


class JasperModel(BaseModel):
    def __init__(self, n_feats, n_class, blocks_num, submodules_num, alpha,
                 beta, prolog_params, blocks_params, epilog_params, **batch):
        super().__init__(n_feats, n_class, **batch)

        self.prolog = JasperSubmodule(in_channels=n_feats, **prolog_params)

        self.blocks = nn.ModuleList([
            JasperBlock(submodules_num=submodules_num, **blocks_params[i]) for i in range(blocks_num)
        ])

        self.epilog = nn.Sequential(
            JasperSubmodule(**epilog_params[0]),
            JasperSubmodule(**epilog_params[1]),
            nn.Conv1d(out_channels=n_class, **epilog_params[2])
        )

        self.alpha, self.beta = alpha, beta

    def forward(self, spectrogram, **batch):
        x = self.prolog(spectrogram)
        for block in self.blocks:
            x = block(x)

        return {"logits": self.epilog(x).transpose(1, 2)}

    def transform_input_lengths(self, input_lengths):
        return (input_lengths * self.alpha + self.beta).type(torch.int)
