import torch
import torch.nn as nn
import torch.nn.functional as F

from hw_tts.model.utils import get_attn_key_pad_mask, get_non_pad_mask, FFTBlock


__all__ = ["LengthRegulator", "Encoder", "Decoder"]



class LengthRegulator(nn.Module):
    """ Length Regulator """
    def __init__(self, duration_predictor_config, predictor):
        super(LengthRegulator, self).__init__()
        self.duration_predictor = predictor(**duration_predictor_config)

    @staticmethod
    def _create_alignment(base_mat, duration_predictor_output):
        n, m = duration_predictor_output.shape
        for i in range(n):
            count = 0
            for j in range(m):
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


class Encoder(nn.Module):
    def __init__(self, num_layers, encoder_dim, encoder_conv1d_filter_size,
                 encoder_head, dropout, max_seq_len, vocab_size, pad=0):
        super(Encoder, self).__init__()

        self.src_word_emb = nn.Embedding(vocab_size, encoder_dim, padding_idx=pad)
        self.position_enc = nn.Embedding(max_seq_len + 1, encoder_dim, padding_idx=pad)

        self.layer_stack = nn.ModuleList([
            FFTBlock(encoder_dim, encoder_conv1d_filter_size, encoder_dim // encoder_head,
                     encoder_dim // encoder_head, encoder_head, dropout=dropout)
            for _ in range(num_layers)])

    def forward(self, src_seq, src_pos, return_attns=False):
        enc_slf_attn_list = []

        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        non_pad_mask = get_non_pad_mask(src_seq)

        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask)

            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        return enc_output, non_pad_mask

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
