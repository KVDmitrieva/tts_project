import torch
import torch.nn as nn

from hw_tts.model.base_model import BaseModel
from hw_tts.model.utils import Transpose, get_mask_from_lengths
from hw_tts.model.common_blocks import *


class DurationPredictor(nn.Module):
    """ Duration Predictor """
    def __init__(self, encoder_dim, duration_predictor_kernel_size,
                 duration_predictor_filter_size, dropout):
        super(DurationPredictor, self).__init__()

        self.conv_net = nn.Sequential(
            Transpose(-1, -2),
            nn.Conv1d(
                encoder_dim, duration_predictor_filter_size,
                kernel_size=duration_predictor_kernel_size, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(duration_predictor_filter_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            Transpose(-1, -2),
            nn.Conv1d(
                duration_predictor_filter_size,
                duration_predictor_filter_size,
                kernel_size=duration_predictor_kernel_size, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(duration_predictor_filter_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.linear_layer = nn.Linear(duration_predictor_filter_size, 1)
        self.relu = nn.ReLU()

    def forward(self, encoder_output):
        encoder_output = self.conv_net(encoder_output)

        out = self.linear_layer(encoder_output)
        out = self.relu(out)
        out = out.squeeze(-1)

        return out


class FastSpeech(BaseModel):
    """ FastSpeech """
    def __init__(self, encoder_params, lr_params, decoder_params, num_mels):
        super().__init__()

        self.encoder = Encoder(**encoder_params)
        self.length_regulator = LengthRegulator(**lr_params, predictor=DurationPredictor)
        self.decoder = Decoder(**decoder_params)

        self.mel_linear = nn.Linear(decoder_params["decoder_dim"], num_mels)

    @staticmethod
    def _mask_tensor(mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = ~get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.)

    def forward(self, text_encoded, src_pos, mel_pos=None, mel_max_length=None, alignment=None, alpha=1.0, **batch):
        x, _ = self.encoder(text_encoded, src_pos)
        if self.training:
            output, duration_predictor_output = self.length_regulator(x, alpha, alignment, mel_max_length)
            output = self.decoder(output, mel_pos)
            output = self._mask_tensor(output, mel_pos, mel_max_length)
            output = self.mel_linear(output)
            return {"mel": output, "duration_predicted": duration_predictor_output}

        output, mel_pos = self.length_regulator(x, alpha)
        output = self.decoder(output, mel_pos)
        output = self.mel_linear(output)
        return output

