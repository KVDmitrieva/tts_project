import torch
import torch.nn as nn
import torch.nn.functional as F

from hw_tts.model.base_model import BaseModel
from hw_tts.model.utils import *
from hw_tts.model.common_blocks import Encoder, FFTBlock


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


class LengthRegulator(nn.Module):
    """ Length Regulator """
    def __init__(self, duration_predictor_config):
        super(LengthRegulator, self).__init__()
        self.duration_predictor = DurationPredictor(**duration_predictor_config)

    @staticmethod
    def _create_alignment(base_mat, duration_predictor_output):
        N, L = duration_predictor_output.shape
        for i in range(N):
            count = 0
            for j in range(L):
                for k in range(duration_predictor_output[i][j]):
                    base_mat[i][count + k][j] = 1
                count = count + duration_predictor_output[i][j]
        return base_mat

    def _regulate_length(self, x, duration_predictor_output, mel_max_length=None):
        expand_max_len = duration_predictor_output.sum(dim=-1).max(dim=-1)[0]
        alignment = torch.zeros(duration_predictor_output.size(0), expand_max_len,
                                duration_predictor_output.size(1)).numpy()
        alignment = self._create_alignment(alignment, duration_predictor_output.cpu())
        alignment = torch.from_numpy(alignment).to(x.device)

        output = alignment @ x
        if mel_max_length:
            output = F.pad(output, (0, 0, 0, mel_max_length - output.size(1), 0, 0))
        return output

    def forward(self, x, alpha=1.0, target=None, mel_max_length=None):
        duration_predictor_output = self.duration_predictor(x)
        if target is not None:
            output = self._regulate_length(x, target, mel_max_length)
            return output, duration_predictor_output

        duration_predictor_output = (alpha * (duration_predictor_output + 0.5)).int()
        output = self._regulate_length(x, duration_predictor_output, mel_max_length)
        mel_pos = torch.arange(1, duration_predictor_output.size(1) + 1).unsqueeze(0).to(x.device)

        return output, mel_pos


class Decoder(nn.Module):
    """ Decoder """

    def __init__(self, num_layers, decoder_dim, decoder_conv1d_filter_size,
                 decoder_head, dropout, max_seq_len, pad=0):

        super(Decoder, self).__init__()

        self.position_enc = nn.Embedding(max_seq_len + 1, decoder_dim, padding_idx=pad)

        self.layer_stack = nn.ModuleList([
            FFTBlock(decoder_dim, decoder_conv1d_filter_size, decoder_dim // decoder_head,
                     decoder_dim // decoder_head, decoder_head, dropout=dropout)
            for _ in range(num_layers)])

    def forward(self, enc_seq, enc_pos, return_attns=False):

        dec_slf_attn_list = []

        slf_attn_mask = get_attn_key_pad_mask(seq_k=enc_pos, seq_q=enc_pos)
        non_pad_mask = get_non_pad_mask(enc_pos)

        dec_output = enc_seq + self.position_enc(enc_pos)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(dec_output, non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask)
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]

        return dec_output


class FastSpeech(BaseModel):
    """ FastSpeech """
    def __init__(self, encoder_params, lr_params, decoder_params, num_mels):
        super().__init__()

        self.encoder = Encoder(**encoder_params)
        self.length_regulator = LengthRegulator(**lr_params)
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
        output, duration_predictor_output = self.length_regulator(x, alpha, alignment, mel_max_length)
        output = self.decoder(output, mel_pos)
        output = self._mask_tensor(output, mel_pos, mel_max_length)
        output = self.mel_linear(output)
        return {"mel": output, "duration_predicted": duration_predictor_output}

    @torch.inference_mode
    def inference(self, text_encoded, src_pos, alpha=1.0, **batch):
        x, _ = self.encoder(text_encoded, src_pos)
        output, mel_pos = self.length_regulator(x, alpha)
        output = self.decoder(output, mel_pos)
        output = self.mel_linear(output)
        return {"mel": output}
