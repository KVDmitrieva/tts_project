import torch
import torch.nn as nn

from hw_tts.model.base_model import BaseModel
from hw_tts.model.utils import *
from hw_tts.model.common_blocks import *


class UniversalPredictor(nn.Module):
    """ Duration/Pitch/Energy Predictor """
    def __init__(self, encoder_dim, duration_predictor_kernel_size,
                 duration_predictor_filter_size, dropout):
        super(UniversalPredictor, self).__init__()

        self.conv_net = nn.Sequential(
            Transpose(-1, -2),
            nn.Conv1d(
                encoder_dim, duration_predictor_filter_size,
                kernel_size=duration_predictor_kernel_size, padding=1
            ),
            Transpose(-1, -2),
            nn.ReLU(),
            nn.LayerNorm(duration_predictor_filter_size),
            nn.Dropout(dropout),
            Transpose(-1, -2),
            nn.Conv1d(
                duration_predictor_filter_size,
                duration_predictor_filter_size,
                kernel_size=duration_predictor_kernel_size, padding=1
            ),
            Transpose(-1, -2),
            nn.ReLU(),
            nn.LayerNorm(duration_predictor_filter_size),
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


class VarianceAdaptor(nn.Module):
    # based on 1st version of paper https://arxiv.org/pdf/2006.04558v1.pdf
    def __init__(self, predictor_params, n_bins=256):
        super().__init__()
        self.length_regulator = LengthRegulator(predictor_params, predictor=UniversalPredictor)
        self.pitch_predictor = UniversalPredictor(**predictor_params)
        self.energy_predictor = UniversalPredictor(**predictor_params)

        self.n_bins = 256
        self.pitch_embedding = nn.Embedding(n_bins, predictor_params["encoder_dim"])
        self.energy_embedding = nn.Embedding(n_bins, predictor_params["encoder_dim"])

    def _extract_pitch(self, x, pitch_alpha=1.0, pitch=None):
        pitch_prediction = self.pitch_predictor(x)
        output = pitch_alpha * pitch_prediction if pitch is None else pitch
        log_out = torch.log(output + 1e-12)
        boundaries = torch.exp(torch.linspace(log_out.min(), log_out.max(), self.n_bins + 1))[1:-1].to(x.device)
        output = self.pitch_embedding(torch.bucketize(output, boundaries, out_int32=True).to(x.device))
        return output, pitch_prediction

    def _extract_energy(self, x, energy_alpha=1.0, energy=None):
        energy_prediction = self.energy_predictor(x)
        output = energy_alpha * energy_prediction if energy is None else energy
        boundaries = torch.linspace(output.min(), output.max(), self.n_bins + 1)[1:-1].to(x.device)
        output = self.energy_embedding(torch.bucketize(output, boundaries, out_int32=True).to(x.device))
        return output, energy_prediction

    def forward(self, x, alpha=1.0, pitch_alpha=1.0, energy_alpha=1.0,
                alignment=None, mel_max_length=None, pitch=None, energy=None, **batch):
        output, duration_predictor_output = self.length_regulator(x, alpha, alignment, mel_max_length)
        pitch_out, pitch_prediction = self._extract_pitch(output, pitch_alpha=pitch_alpha, pitch=pitch)
        energy_out, energy_prediction = self._extract_energy(output, energy_alpha=energy_alpha, energy=energy)
        output += pitch_out + energy_out
        print(output.dtype, duration_predictor_output.dtype, pitch_prediction.dtype, energy_prediction.dtype)
        return output, duration_predictor_output, pitch_prediction, energy_prediction


class FastSpeech2(BaseModel):
    """ FastSpeech 2"""
    def __init__(self, encoder_params, va_params, decoder_params, num_mels):
        super().__init__()

        self.encoder = Encoder(**encoder_params)
        self.variance = VarianceAdaptor(**va_params)
        self.decoder = Decoder(**decoder_params)

        self.mel_linear = nn.Linear(decoder_params["decoder_dim"], num_mels)

    @staticmethod
    def _mask_tensor(mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = ~get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.)

    def forward(self, text_encoded, src_pos, mel_pos=None, mel_max_length=None, alpha=1.0, pitch_alpha=1.0, energy_alpha=1.0, **batch):
        x, _ = self.encoder(text_encoded, src_pos)
        output, duration_out, pitch_out, energy_out = self.variance(x, alpha, mel_max_length=mel_max_length, **batch)
        output = self.decoder(output, mel_pos)
        output = self._mask_tensor(output, mel_pos, mel_max_length)
        output = self.mel_linear(output)
        return {"mel": output, "duration_predicted": duration_out,
                "pitch_predicted": pitch_out, "energy_predicted": energy_out}

    @torch.inference_mode()
    def inference(self, text_encoded, src_pos, alpha=1.0, pitch_alpha=1.0, energy_alpha=1.0, **batch):
        self.eval()
        x, _ = self.encoder(text_encoded, src_pos)
        output, mel_pos, _, _ = self.variance(x, alpha, pitch_alpha, energy_alpha)
        output = self.decoder(output, mel_pos)
        output = self.mel_linear(output)
        return output
