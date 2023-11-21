import logging
import torch
from typing import List
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    mel, text_encoded = [], []
    text, text_encoded_length = [], []
    alignment, audio = [], []
    src_pos, mel_pos = [], []
    mel_max_len = 0

    for item in dataset_items:
        if "mel" in item:
            alignment.append(item["alignment"])
            mel.append(item["mel"].unsqueeze(0).transpose(1, 2))
            mel_max_len = max(mel_max_len, item["mel"].shape[-1])
            mel_pos.append(torch.arange(1, item["mel"].shape[-1] + 1))

        text.append(item["text"])
        text_encoded.append(item["text_encoded"].squeeze(0))
        src_pos.append(torch.arange(1, text_encoded_length[-1] + 1))

    return {
        "text": text,
        "alignment": alignment if len(mel) > 0 else None,
        "src_pos": pad_sequence(src_pos, batch_first=True),
        "mel_max_len": alignment if len(mel) > 0 else None,
        "text_encoded": pad_sequence(text_encoded, batch_first=True),
        "mel_pos": pad_sequence(mel_pos, batch_first=True) if len(mel) > 0 else None,
        "mel_target":  pad_sequence(mel, batch_first=True).transpose(1, 2) if len(mel) > 0 else None
    }
