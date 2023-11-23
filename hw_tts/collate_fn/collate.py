import logging
import torch
from typing import List
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    text, text_encoded = [], []
    mel, mel_len = [], []
    alignment, audio = [], []
    src_pos, mel_pos = [], []
    pitch, energy = [], []
    mel_len = []
    mel_max_len = 0

    for item in dataset_items:
        mel.append(item["mel"])
        text.append(item["text"])
        energy.append(item["energy"])
        mel_len.append(item["mel"].shape[0])
        pitch.append(torch.tensor(item["pitch"]))
        alignment.append(torch.tensor(item["alignment"]))
        mel_max_len = max(mel_max_len, item["mel"].shape[0])
        text_encoded.append(torch.tensor(item["text_encoded"]))
        mel_pos.append(torch.arange(1, item["mel"].shape[0] + 1))
        src_pos.append(torch.arange(1, len(item["text_encoded"]) + 1))

    return {
        "text": text,
        "mel_len": mel_len,
        "mel_max_len": mel_max_len,
        "pitch": pad_sequence(pitch, batch_first=True),
        "energy": pad_sequence(energy, batch_first=True),
        "mel_pos": pad_sequence(mel_pos, batch_first=True).long(),
        "src_pos": pad_sequence(src_pos, batch_first=True).long(),
        "mel_target": pad_sequence(mel, batch_first=True).float(),
        "alignment": pad_sequence(alignment, batch_first=True).int(),
        "text_encoded": pad_sequence(text_encoded, batch_first=True).long()
    }
