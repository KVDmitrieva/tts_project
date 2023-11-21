import numpy as np
import torch.nn.functional as F

from hw_tts.model.base_model import BaseModel
from hw_tts.model.utils import *


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        scaled_dot = torch.bmm(q, k.transpose(1, 2))
        scaled_dot /= self.temperature
        if mask is not None:
            scaled_dot = scaled_dot.masked_fill(mask == 0, -torch.inf)

        attn = self.softmax(scaled_dot)
        attn = self.dropout(attn)

        output = torch.bmm(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)

        self.attention = ScaledDotProductAttention(
            temperature=d_k**0.5)
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
         # normal distribution initialization better than kaiming(default in pytorch)
        nn.init.normal_(self.w_qs.weight, mean=0,
                        std=np.sqrt(2.0 / (self.d_model + self.d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0,
                        std=np.sqrt(2.0 / (self.d_model + self.d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0,
                        std=np.sqrt(2.0 / (self.d_model + self.d_v)))

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class PositionWiseFeedForward(nn.Module):
    """A two-feed-forward-layer module"""

    def __init__(self, embed_dim, hidden_dim, fft_conv1d_kernel=(9, 1), fft_conv1d_padding=(4, 0), dropout=0.1):
        super().__init__()

        # position-wise
        self.w_1 = nn.Conv1d(embed_dim, hidden_dim, kernel_size=fft_conv1d_kernel[0], padding=fft_conv1d_padding[0])
        self.w_2 = nn.Conv1d(hidden_dim, embed_dim, kernel_size=fft_conv1d_kernel[1], padding=fft_conv1d_padding[1])

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)

        return output


class FFTBlock(nn.Module):
    """FFT Block"""

    def __init__(self, embed_dim, hidden_dim, k_dim, v_dim, num_heads, dropout=0.1):
        super(FFTBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head=num_heads, d_model=embed_dim, d_k=k_dim, d_v=v_dim, dropout=dropout)
        self.pos_ffn = PositionWiseFeedForward(embed_dim, hidden_dim, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)

        if non_pad_mask is not None:
            enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)

        if non_pad_mask is not None:
            enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


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
