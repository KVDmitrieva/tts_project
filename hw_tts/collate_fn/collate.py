import logging
import torch
from typing import List
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    text, mel, text_encoded = [], [], []
    alignment, audio = [], []
    src_pos, mel_pos = [], []
    mel_len = []
    mel_max_len = 0

    for item in dataset_items:
        text.append(item["text"])
        mel_len.append(item["mel"].shape[0])
        mel.append(torch.tensor(item["mel"]))
        alignment.append(torch.tensor(item["alignment"]))
        mel_max_len = max(mel_max_len, item["mel"].shape[0])
        text_encoded.append(torch.tensor(item["text_encoded"]))
        mel_pos.append(torch.arange(1, item["mel"].shape[0] + 1))
        src_pos.append(torch.arange(1, len(item["text_encoded"]) + 1))

    return {
        "text": text,
        "mel_len": mel_len,
        "mel_max_len": mel_max_len,
        "mel_pos": pad_sequence(mel_pos, batch_first=True).long(),
        "src_pos": pad_sequence(src_pos, batch_first=True).long(),
        "alignment": pad_sequence(alignment, batch_first=True).int(),
        "text_encoded": pad_sequence(text_encoded, batch_first=True).long(),
        "mel_target":  pad_sequence(mel, batch_first=True).float()
    }
