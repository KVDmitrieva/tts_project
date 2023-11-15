import logging
import torch
from typing import List

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    spectrogram, text_encoded = [], []
    text, text_encoded_length = [], []
    spectrogram_length, audio_path = [], []

    for item in dataset_items:
        text.append(item["text"])
        audio_path.append(item["audio_path"])
        spectrogram.append(item["spectrogram"].squeeze(0).T)
        text_encoded.append(item["text_encoded"].squeeze(0))
        spectrogram_length.append(item["spectrogram"].shape[2])
        text_encoded_length.append(item["text_encoded"].shape[1])

    return {
        "text": text,
        "audio": item["audio"],
        "audio_path": audio_path,
        "spectrogram_length": torch.tensor(spectrogram_length),
        "text_encoded_length": torch.tensor(text_encoded_length),
        "text_encoded": torch.nn.utils.rnn.pad_sequence(text_encoded, batch_first=True),
        "spectrogram":  torch.nn.utils.rnn.pad_sequence(spectrogram, batch_first=True).transpose(1, 2)
    }