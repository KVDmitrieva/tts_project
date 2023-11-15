from typing import List, NamedTuple
from collections import defaultdict
from pyctcdecode import build_ctcdecoder

import torch
import numpy as np

from .char_text_encoder import CharTextEncoder


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None, lm_path=None, alpha=0.5, beta=1):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

        if lm_path is not None:
            self.decoder = build_ctcdecoder(
                [''] + [ch.upper() for ch in self.alphabet],
                kenlm_model_path=lm_path,
                alpha=alpha,
                beta=beta,
            )

    def ctc_decode(self, inds: List[int]) -> str:
        result = []
        last_char = self.EMPTY_TOK
        for ind in inds:
            next_char = self.ind2char[ind]
            if next_char != self.EMPTY_TOK and next_char != last_char:
                result.append(next_char)

            last_char = next_char

        return ''.join(result)

    def ctc_lm_beam_search(self, log_probs: torch.tensor, log_probs_length,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(log_probs.shape) == 2
        char_length, voc_size = log_probs.shape
        assert voc_size == len(self.ind2char)
        hypos: List[Hypothesis] = []

        probs = log_probs[:log_probs_length]
        beam_pred = self.decoder.decode_beams(probs, beam_width=beam_size)
        for text, _, _, _, lm_log_prob in beam_pred:
            hypos.append(Hypothesis(text.lower(), np.exp(lm_log_prob)))
        return sorted(hypos, key=lambda x: x.prob, reverse=True)

    def ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        hypos: List[Hypothesis] = [Hypothesis('', 1.0)]

        probs = probs[:probs_length]
        for frame in probs:
            hypos = self._extend_and_merge(frame, hypos)
            hypos = self._truncate(hypos, beam_size)
        return sorted(hypos, key=lambda x: x.prob, reverse=True)

    def _extend_and_merge(self, frame, hypos):
        new_hypos = defaultdict(float)
        for next_char_index, next_char_proba in enumerate(frame):
            for pref, pref_proba in hypos:
                new_pref = pref
                next_char = self.ind2char[next_char_index]
                last_char = pref[-1] if pref else self.EMPTY_TOK
                if next_char != self.EMPTY_TOK and next_char != last_char:
                    new_pref += next_char
                new_hypos[new_pref] += next_char_proba * pref_proba

        return [Hypothesis(text, probs) for text, probs in new_hypos.items()]

    @staticmethod
    def _truncate(hypos, beam_size):
        sorted_hypos = sorted(hypos, key=lambda x: x.prob, reverse=True)
        return sorted_hypos[:beam_size]
