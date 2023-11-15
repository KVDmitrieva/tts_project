from typing import List

import torch
from torch import Tensor

from hw_tts.metric.base_metric import BaseMetric
from hw_tts.text_encoder.base_text_encoder import BaseTextEncoder
from hw_tts.metric.utils import calc_cer


class ArgmaxCERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs):
        cers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        lengths = log_probs_length.detach().numpy()
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = BaseTextEncoder.normalize_text(target_text)
            if hasattr(self.text_encoder, "ctc_decode"):
                pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length])
            else:
                pred_text = self.text_encoder.decode(log_prob_vec[:length])
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)


class BeamSearchCERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], beam_size=5, **kwargs):
        cers = []
        lengths = log_probs_length.detach().numpy()
        for log_prob, length, target_text in zip(log_probs, lengths, text):
            pred_text = self.text_encoder.ctc_beam_search(torch.exp(log_prob), length, beam_size)[0].text
            target_text = BaseTextEncoder.normalize_text(target_text)
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)


class LMBeamSearchCERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], beam_size=5, **kwargs):
        cers = []
        lengths = log_probs_length.detach().numpy()
        log_probs = log_probs.cpu().detach().numpy()
        for log_prob, length, target_text in zip(log_probs, lengths, text):
            pred_text = self.text_encoder.ctc_lm_beam_search(log_prob, length, beam_size)[0].text
            target_text = BaseTextEncoder.normalize_text(target_text)
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)
